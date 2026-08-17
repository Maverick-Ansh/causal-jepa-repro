"""Is the VQA probe strong enough to be a measuring instrument?

The probe is only useful if it can tell a good imagined future from a bad one.
The decisive check is the gap between two arms that differ ONLY in the trajectory:

    oracle : probe reads the TRUE future latents   (ceiling)
    static : probe reads the last observed frame repeated  (floor, no dynamics)

If ceiling ~= floor, the metric is blind and no world-model comparison built on
it means anything. Run this before committing GPU hours to a sweep.

    python -m scripts.check_probe --steps 1200 --test_eps 1536
"""
import argparse
import time

import torch

from cjepa.data import SplitSpec, build_world_data
from cjepa.encoders import build_encoder
from cjepa.envs import InteractionWorld, WorldConfig
from cjepa.eval.qa import QTYPES, build_qa_bank, imagined_trajectories, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--test_eps", type=int, default=1536)
ap.add_argument("--cache", default="/kaggle/working/cache")
args = ap.parse_args()

dev = "cuda"
Th, Tp, skip = 6, 10, 2
w = WorldConfig(n_frames=64)
env = InteractionWorld(w, device=dev)

data = build_world_data(w, {"train": SplitSpec(2048, 0),
                            "test": SplitSpec(args.test_eps, 555)},
                        Th, Tp, skip, dev, args.cache)
enc = build_encoder("oracle", w.state_dim(), 128, 0).to(dev)
enc.fit_normalizer(data["train"].states)

t0 = time.time()
bank = build_qa_bank(env, data["test"].rollout, Th_sim=Th * skip,
                     T_sim=(Th + Tp) * skip, n_per_type=4, seed=0)
tr, te = bank.split(0.75, seed=0)
print(f"[qa] {bank.ep.numel()} questions (train {tr.ep.numel()}/eval {te.ep.numel()}) "
      f"in {time.time()-t0:.0f}s")
print("[qa] copy-factual baseline: " +
      "  ".join(f"{k} {v:.1f}" for k, v in bank.copy_factual_baseline().items()))

ep = torch.arange(data["test"].B, device=dev)
st = torch.zeros(data["test"].B, device=dev, dtype=torch.long)
states = data["test"].window_states(ep, st)

res = {}
for mode in ["oracle", "static"]:
    t0 = time.time()
    traj = imagined_trajectories(None, enc, states, Th, mode=mode)
    res[mode] = train_probe(traj, tr, te, steps=args.steps, seed=0, device=dev)
    print(f"[{mode:6s}] " + "  ".join(f"{k} {v:5.1f}" for k, v in res[mode].items()
                                      if k != "n_eval") + f"   ({time.time()-t0:.0f}s)")
    del traj
    torch.cuda.empty_cache()

print("\nceiling - floor (this is the headroom any world-model comparison lives in):")
worst = 1e9
for c in QTYPES + ["average"]:
    d = res["oracle"][c] - res["static"][c]
    print(f"  {c:15s} {res['oracle'][c]:5.1f} - {res['static'][c]:5.1f} = {d:+5.1f}")
    if c in ("predictive", "counterfactual"):
        worst = min(worst, d)
print(f"\ndynamics-sensitive headroom (min of predictive/counterfactual): {worst:+.1f} pts")
print("VERDICT:", "USABLE" if worst >= 5.0 else "TOO WEAK — do not run the sweep")
