"""
Reproduce the planning-efficiency claim (Sec. 5.2 / Tab. 3).

THE CLAIM
---------
Tab. 3 reports token budgets of  196 x 384  (DINO-WM, patch-based) versus
6 x 128 (object-centric), and the text says:

    "Despite using only 1.02% of the latent tokens, it achieves performance
     comparable to patch-based world models. Because predictor rollouts dominate
     the computational cost in model-based planning, this reduction directly
     translates to efficiency gains. Under identical settings on a single L40s
     GPU, C-JEPA achieves over 8x faster planning, requiring 673 seconds on
     average across three seeds to evaluate 50 trajectories, compared to 5,763
     seconds for DINO-WM."

WHERE 1.02% COMES FROM (worth stating, because it pins down the tokenisation)
    6 * 128 / (196 * 384) = 768 / 75264 = 1.0204%
The "6" is 4 object slots (App. C.2: "we use four object-centric slots") plus one
action token plus one proprioception token — auxiliaries are separate entities in
C-JEPA (Sec. 5.2, Fig. 3), not concatenated into the visual features.

WHY THIS ARM IS CHEAP AND STILL HONEST
--------------------------------------
Success rate (the other half of Tab. 3) needs the full Push-T environment,
demonstration data, and trained world models — out of scope here, and we say so
in REPORT.md rather than inventing numbers. But the *efficiency* half is
determined entirely by predictor cost under a fixed CEM workload, and both arms
of the paper's comparison use the SAME predictor architecture:

    App. E.2  (C-JEPA):  6 layers, 16 heads, head dim 64, MLP 2048
    App. H    (DINO-WM): 6 layers, 16 heads, MLP 2048, per-head dim 64

So the only thing that differs is sequence length: 4 timesteps x 196 patches
versus 4 timesteps x 6 entities. That makes the speedup measurable exactly, with
no training and no environment.

CEM WORKLOAD (App. G, verbatim)
    planning horizon H = 5, action block B = 5, L = H*B = 25
    receding horizon R = 5, 50 environment steps per episode -> 10 replans
    CEM: 300 candidate sequences, 30 elites, 30 iterations
    => per replan  : 30 iters x 300 candidates x H=5 predictor steps
    => per episode : 10 replans
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from cjepa.models.predictor import CJEPAPredictor, PredictorConfig

# --- App. G ---------------------------------------------------------------- #
CEM_SAMPLES = 300
CEM_ITERS = 30
PLAN_H = 5
ENV_STEPS = 50
RECEDING = 5
REPLANS_PER_EPISODE = ENV_STEPS // RECEDING          # 10


def bench_predictor(n_tokens_per_step: int, slot_dim: int, Th: int, Tp: int,
                    d_model: int = 1024, n_heads: int = 16, mlp: int = 2048,
                    n_layers: int = 6, batch: int = CEM_SAMPLES,
                    iters: int = 30, device: str = "cuda", fp16: bool = True):
    """Time one predictor forward pass at the given token budget."""
    cfg = PredictorConfig(slot_dim=slot_dim, d_model=d_model, n_heads=n_heads,
                          head_dim=d_model // n_heads, n_layers=n_layers,
                          mlp_hidden=mlp, Th=Th, Tp=Tp,
                          n_slots=n_tokens_per_step)
    model = CJEPAPredictor(cfg).to(device).eval()
    T = Th + Tp
    z = torch.randn(batch, T, n_tokens_per_step, slot_dim, device=device)
    mask = torch.zeros(batch, T, n_tokens_per_step, dtype=torch.bool, device=device)
    mask[:, Th:] = True

    with torch.no_grad():
        for i in range(iters + 5):
            if i == 5:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                model(z, mask)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    n_params = sum(p.numel() for p in model.parameters())
    del model, z, mask
    torch.cuda.empty_cache()
    return dt, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/results/planning_cost.json")
    ap.add_argument("--episodes", type=int, default=50)   # Tab. 3 evaluates 50
    args = ap.parse_args()

    arms = {
        # name           tokens/step  slot_dim  Th  Tp
        "DINO-WM (patch)":  (196, 384, 3, 1),
        "C-JEPA (object)":  (6,   128, 3, 1),
    }

    results = {}
    for name, (ntok, sd, Th, Tp) in arms.items():
        dt, npar = bench_predictor(ntok, sd, Th, Tp)
        # one CEM forward evaluates all 300 candidates at once, repeated over
        # the planning horizon and the CEM iterations
        per_replan = dt * CEM_ITERS * PLAN_H
        per_episode = per_replan * REPLANS_PER_EPISODE
        total = per_episode * args.episodes
        results[name] = {
            "tokens_per_step": ntok,
            "feature_dim": sd,
            "input_feature_size": ntok * sd,
            "seq_len": ntok * (Th + Tp),
            "predictor_params_M": npar / 1e6,
            "fwd_ms_batch300": dt * 1e3,
            "sec_per_replan": per_replan,
            "sec_per_episode": per_episode,
            f"sec_for_{args.episodes}_episodes": total,
        }
        print(f"{name:20s} tokens {ntok:4d}x{sd:<4d} = {ntok*sd:6d} feat  "
              f"fwd {dt*1e3:7.2f} ms  -> {total:8.1f} s for {args.episodes} episodes")

    a, b = results["DINO-WM (patch)"], results["C-JEPA (object)"]
    k = f"sec_for_{args.episodes}_episodes"
    results["summary"] = {
        "input_feature_ratio_pct": 100 * b["input_feature_size"] / a["input_feature_size"],
        "planning_speedup": a[k] / b[k],
        "paper_input_feature_ratio_pct": 1.02,
        "paper_speedup": 5763 / 673,
        "paper_dinowm_sec": 5763,
        "paper_cjepa_sec": 673,
        "cem": {"samples": CEM_SAMPLES, "iters": CEM_ITERS, "horizon": PLAN_H,
                "replans_per_episode": REPLANS_PER_EPISODE, "episodes": args.episodes},
    }
    s = results["summary"]
    print(f"\ninput feature ratio : {s['input_feature_ratio_pct']:.2f}%  "
          f"(paper: {s['paper_input_feature_ratio_pct']}%)")
    print(f"planning speedup    : {s['planning_speedup']:.2f}x  "
          f"(paper: {s['paper_speedup']:.2f}x on an L40s)")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
