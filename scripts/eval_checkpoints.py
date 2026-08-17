"""
Second evaluation pass over saved checkpoints: the probe-free measurements.

    python -m scripts.eval_checkpoints --sweep results/sweep

Adds to each run's JSON:
  * counterfactual rollout metrics (cjepa/eval/counterfactual.py)
  * collision-prediction F1 over the imagined horizon

These are deliberately run *after* training rather than inline, so they can be
re-run or extended without retraining anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from cjepa.data import SplitSpec, build_world_data
from cjepa.encoders import build_encoder
from cjepa.envs import InteractionWorld, WorldConfig
from cjepa.eval.counterfactual import collision_prediction, counterfactual_rollout
from cjepa.models.predictor import CJEPAPredictor, PredictorConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="/kaggle/working/results/sweep")
    ap.add_argument("--cache", default="/kaggle/working/cache")
    ap.add_argument("--test_eps", type=int, default=1536)
    ap.add_argument("--n_cf", type=int, default=384)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    args = ap.parse_args()

    dev = "cuda"
    Th, Tp, skip = 6, 10, 2
    wcfg = WorldConfig(n_frames=64)
    env = InteractionWorld(wcfg, device=dev)

    data = build_world_data(
        wcfg, {"train": SplitSpec(2048, 0), "test": SplitSpec(args.test_eps, 555)},
        Th, Tp, frame_skip=skip, device=dev, cache_dir=args.cache)

    encs = {}
    for kind in ["oracle", "degraded"]:
        e = build_encoder(kind, wcfg.state_dim(), 128, seed=0).to(dev)
        e.fit_normalizer(data["train"].states)
        encs[kind] = e

    B = min(args.n_cf, data["test"].B)
    ep = torch.arange(B, device=dev)
    st = torch.zeros(B, device=dev, dtype=torch.long)
    states = data["test"].window_states(ep, st)

    ckpts = sorted(glob.glob(os.path.join(args.sweep, "*.pt")))
    print(f"found {len(ckpts)} checkpoints")
    for cp in ckpts:
        tag = os.path.basename(cp)[:-3]
        jf = os.path.join(args.sweep, f"{tag}.json")
        if not os.path.exists(jf):
            print(f"skip {tag}: no json"); continue
        res = json.load(open(jf))
        if "counterfactual" in res:
            print(f"skip {tag}: already evaluated"); continue

        enc_kind = res.get("encoder") or ("degraded" if "degraded" in tag else "oracle")
        pcfg = PredictorConfig(slot_dim=128, d_model=args.d_model,
                               n_heads=max(1, args.d_model // 64), head_dim=64,
                               n_layers=6, mlp_hidden=args.mlp_hidden,
                               Th=Th, Tp=Tp, n_slots=wcfg.n_slots)
        model = CJEPAPredictor(pcfg).to(dev)
        model.load_state_dict(torch.load(cp, map_location=dev))

        res["counterfactual"] = counterfactual_rollout(
            model, encs[enc_kind], env, states, frame_skip=skip)
        res["collision"] = collision_prediction(model, encs[enc_kind], states)
        json.dump(res, open(jf, "w"), indent=2)
        c, k = res["counterfactual"], res["collision"]
        print(f"{tag:24s} cf_gain {c['cf_gain']:+.3f} (n={c['n']}) "
              f"fwd_err {c['fwd_err']:.4f}  coll_F1 {k['collision_f1']:.3f} "
              f"(P {k['collision_precision']:.3f} R {k['collision_recall']:.3f})")
        del model
        torch.cuda.empty_cache()
    print("EVAL_CHECKPOINTS DONE")


if __name__ == "__main__":
    main()
