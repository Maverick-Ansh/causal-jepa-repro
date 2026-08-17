"""
The main experiment: reproduce Tab. 1 — accuracy as a function of the object
masking budget |M|, for a strong and a weak frozen encoder.

    python -m scripts.run_sweep --shard 0 --n_shards 2 --out results/sweep

Each run trains a predictor and then immediately evaluates it, writing one JSON:

  * val latent MSE / position error        (forward prediction quality)
  * VQA accuracy by CLEVRER category       (Tab. 1's metric, via the ALOE probe)
  * influence-neighborhood alignment       (Cor. 1, our addition)

Grid
----
  encoder  in {oracle, degraded}     <- stands in for {VideoSAUR, SAVi}
  |M|      in {0, 1, 2, 3, 4}        <- Tab. 1 rows; |M| = 0 IS OC-JEPA
  seed     in {42, 43}               <- App. H uses seed 42; we add one more
                                        because Tab. 1 reports single runs and
                                        some of its deltas are ~2 points.

DEVIATIONS FROM THE PAPER, ALL FORCED BY A 2xT4 BUDGET (see REPORT.md):
  * predictor width 256 not 1024 (6 layers / head_dim 64 / MLP shape kept)
  * 2 seeds, and a synthetic world in place of CLEVRER
Everything under test — the masking rule, the loss, the anchor, the
inference-time protocol — is the paper's.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import torch

from cjepa.data import SplitSpec, build_world_data
from cjepa.encoders import build_encoder
from cjepa.envs import WorldConfig, InteractionWorld
from cjepa.eval.influence import evaluate_influence
from cjepa.eval.qa import build_qa_bank, imagined_trajectories, train_probe
from cjepa.models.predictor import PredictorConfig
from cjepa.train import TrainConfig, train_cjepa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/results/sweep")
    ap.add_argument("--cache", default="/kaggle/working/cache")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)      # App. E.3
    ap.add_argument("--train_eps", type=int, default=2048)
    ap.add_argument("--val_eps", type=int, default=512)
    ap.add_argument("--test_eps", type=int, default=1536)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--masks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--encoders", nargs="+", default=["oracle", "degraded"])
    ap.add_argument("--count_mode", default="uniform_upto")
    ap.add_argument("--strategy", default="object")
    ap.add_argument("--anchor", type=int, default=1)
    ap.add_argument("--probe_steps", type=int, default=1500)
    ap.add_argument("--infl_mask", type=int, default=2)
    ap.add_argument("--tag_prefix", default="")
    ap.add_argument("--skip_qa", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"shard{args.shard}.log")

    def log(m):
        print(m, flush=True)
        with open(log_path, "a") as fh:
            fh.write(m + "\n")

    # ---- CLEVRER-shaped temporal setup (App. E.3) --------------------------- #
    Th, Tp, skip = 6, 10, 2
    wcfg = WorldConfig(n_frames=64)

    data = build_world_data(
        wcfg,
        {"train": SplitSpec(args.train_eps, 0),
         "val": SplitSpec(args.val_eps, 777),
         "test": SplitSpec(args.test_eps, 555)},
        Th, Tp, frame_skip=skip, device=dev, cache_dir=args.cache,
    )

    encs = {}
    for kind in args.encoders:
        e = build_encoder(kind, wcfg.state_dim(), 128, seed=0).to(dev)
        e.fit_normalizer(data["train"].states)      # frozen after this
        encs[kind] = e

    # ---- QA benchmark, built ONCE and shared by every world model ----------- #
    qa_tr = qa_te = test_states = None
    if not args.skip_qa:
        t0 = time.time()
        env = InteractionWorld(wcfg, device=dev)
        bank = build_qa_bank(env, data["test"].rollout,
                             Th_sim=Th * skip, T_sim=(Th + Tp) * skip,
                             n_per_type=4, seed=0)
        qa_tr, qa_te = bank.split(0.75, seed=0)
        ep = torch.arange(data["test"].B, device=dev)
        st = torch.zeros(data["test"].B, device=dev, dtype=torch.long)
        test_states = data["test"].window_states(ep, st)
        log(f"[qa] {bank.ep.numel()} questions "
            f"(train {qa_tr.ep.numel()} / eval {qa_te.ep.numel()}) "
            f"built in {time.time()-t0:.0f}s")
        for t, name in enumerate(["descriptive", "predictive",
                                  "counterfactual", "explanatory"]):
            m = bank.qtype == t
            log(f"[qa]   {name:15s} n={int(m.sum()):6d} "
                f"P(yes)={float(bank.answer[m].float().mean()):.3f}")

    # ---- controls: ceiling and floor, computed once per encoder ------------- #
    if not args.skip_qa and args.shard == 0:
        for kind in args.encoders:
            for mode in ["oracle", "static"]:
                tagc = f"CONTROL_{kind}_{mode}"
                fp = os.path.join(args.out, f"{tagc}.json")
                if os.path.exists(fp):
                    continue
                traj = imagined_trajectories(None, encs[kind], test_states, Th, mode=mode)
                acc = train_probe(traj, qa_tr, qa_te, steps=args.probe_steps,
                                  seed=0, device=dev)
                json.dump({"tag": tagc, "control": mode, "encoder": kind,
                           "vqa": acc}, open(fp, "w"), indent=2)
                log(f"[control] {tagc}: {acc}")
                del traj
                torch.cuda.empty_cache()

    grid = list(itertools.product(args.encoders, args.masks, args.seeds))
    mine = [c for i, c in enumerate(grid) if i % args.n_shards == args.shard]
    log(f"=== shard {args.shard}/{args.n_shards}: {len(mine)} of {len(grid)} runs ===")

    for enc_kind, n_mask, seed in mine:
        tag = f"{args.tag_prefix}{enc_kind}_M{n_mask}_s{seed}"
        if args.strategy != "object":
            tag = f"{args.tag_prefix}{args.strategy}_{enc_kind}_M{n_mask}_s{seed}"
        if not args.anchor:
            tag += "_noanchor"
        if os.path.exists(os.path.join(args.out, f"{tag}.json")):
            log(f"skip {tag} (already done)")
            continue

        pcfg = PredictorConfig(
            slot_dim=128, d_model=args.d_model, n_heads=max(1, args.d_model // 64),
            head_dim=64, n_layers=6, mlp_hidden=args.mlp_hidden,
            Th=Th, Tp=Tp, n_slots=wcfg.n_slots,
        )
        tcfg = TrainConfig(
            n_mask=n_mask, mask_strategy=args.strategy, count_mode=args.count_mode,
            anchor=bool(args.anchor), epochs=args.epochs, batch_size=256, lr=5e-4,
            seed=seed, tag=tag,
        )
        t0 = time.time()
        model, result = train_cjepa(pcfg, tcfg, encs[enc_kind], data["train"],
                                    data["val"], device=dev, log_path=log_path,
                                    out_dir=None)
        t_train = time.time() - t0

        # ---- VQA (Tab. 1's metric) ----------------------------------------- #
        if not args.skip_qa:
            traj = imagined_trajectories(model, encs[enc_kind], test_states, Th,
                                         mode="model")
            result["vqa"] = train_probe(traj, qa_tr, qa_te, steps=args.probe_steps,
                                        seed=0, device=dev)
            del traj
            torch.cuda.empty_cache()
            log(f"[{tag}] VQA {result['vqa']}")

        # ---- influence neighborhoods (Cor. 1) ------------------------------- #
        result["influence"] = evaluate_influence(
            model, encs[enc_kind], data["test"], data["test"].rollout,
            n_mask=args.infl_mask, n_episodes=192, frame_skip=skip, seed=0)
        log(f"[{tag}] influence {result['influence']}")

        result["encoder"] = enc_kind
        result["n_mask"] = n_mask
        result["seed"] = seed
        result["t_train_s"] = t_train
        json.dump(result, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
        torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
        log(f"done {tag} in {time.time()-t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    log(f"=== shard {args.shard} COMPLETE ===")


if __name__ == "__main__":
    main()
