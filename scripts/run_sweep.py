"""
The main experiment: reproduce Tab. 1 — VQA / prediction quality as a function of
the object masking budget |M|, for a strong and a weak frozen encoder.

    python -m scripts.run_sweep --shard 0 --n_shards 2 --out results/sweep

Grid
----
  encoder  in {oracle, degraded}     <- stands in for {VideoSAUR, SAVi}
  |M|      in {0, 1, 2, 3, 4}        <- Tab. 1 rows; |M| = 0 IS OC-JEPA
  seed     in {42, 43, 44}           <- App. H uses seed 42; we add two more
                                        because the effect sizes we care about
                                        (Tab. 1 SAVi rows move by ~2 points) are
                                        not obviously bigger than seed noise, and
                                        the paper reports single runs.

Sharding lets us run one process per T4 (`--shard i --n_shards 2`).
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
from cjepa.envs import WorldConfig
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
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--masks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--encoders", nargs="+", default=["oracle", "degraded"])
    ap.add_argument("--count_mode", default="uniform_upto")
    ap.add_argument("--strategy", default="object")
    ap.add_argument("--anchor", type=int, default=1)
    ap.add_argument("--tag_prefix", default="")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"shard{args.shard}.log")

    # ---- CLEVRER-shaped temporal setup (App. E.3) --------------------------- #
    Th, Tp, skip = 6, 10, 2
    wcfg = WorldConfig(n_frames=64)

    data = build_world_data(
        wcfg,
        {"train": SplitSpec(args.train_eps, 0), "val": SplitSpec(args.val_eps, 777)},
        Th, Tp, frame_skip=skip, device=dev, cache_dir=args.cache,
    )

    # encoders are frozen; fit the standardiser on TRAIN states only
    encs = {}
    for kind in args.encoders:
        e = build_encoder(kind, wcfg.state_dim(), 128, seed=0).to(dev)
        e.fit_normalizer(data["train"].states)
        encs[kind] = e

    grid = list(itertools.product(args.encoders, args.masks, args.seeds))
    mine = [c for i, c in enumerate(grid) if i % args.n_shards == args.shard]
    with open(log_path, "a") as fh:
        fh.write(f"=== shard {args.shard}/{args.n_shards}: {len(mine)} runs ===\n")
    print(f"shard {args.shard}: {len(mine)} of {len(grid)} runs", flush=True)

    for enc_kind, n_mask, seed in mine:
        tag = f"{args.tag_prefix}{enc_kind}_M{n_mask}_s{seed}"
        if args.strategy != "object":
            tag = f"{args.tag_prefix}{args.strategy}_{enc_kind}_M{n_mask}_s{seed}"
        if not args.anchor:
            tag += "_noanchor"
        if os.path.exists(os.path.join(args.out, f"{tag}.json")):
            print(f"skip {tag} (done)", flush=True)
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
        train_cjepa(pcfg, tcfg, encs[enc_kind], data["train"], data["val"],
                    device=dev, log_path=log_path, out_dir=args.out)
        print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)

    with open(log_path, "a") as fh:
        fh.write(f"=== shard {args.shard} COMPLETE ===\n")
    print(f"SHARD {args.shard} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
