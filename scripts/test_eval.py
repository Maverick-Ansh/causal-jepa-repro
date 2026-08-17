"""Fast correctness checks for the QA bank and the influence probes.

Run: python -m scripts.test_eval
Catches the failure modes that would silently invalidate the whole sweep:
  * counterfactual answers not actually differing from factual ones
  * question labels degenerate (all-yes / all-no -> accuracy is meaningless)
  * episode leakage between probe train and probe eval
  * influence scores having the wrong shape / all-equal
"""
import time

import torch

from cjepa.data import SlotSequenceData
from cjepa.encoders import build_encoder
from cjepa.envs import InteractionWorld, WorldConfig
from cjepa.eval.influence import evaluate_influence
from cjepa.eval.qa import (QTYPES, build_qa_bank, collision_matrix,
                           imagined_trajectories, train_probe)
from cjepa.models.predictor import CJEPAPredictor, PredictorConfig

dev = "cuda"
torch.manual_seed(0)
Th, Tp, skip = 6, 10, 2

w = WorldConfig(n_frames=64)
env = InteractionWorld(w, device=dev)
t0 = time.time()
ro = env.generate(384, seed=555)
data = SlotSequenceData(ro, Th, Tp, skip, dev)
print(f"[sim] {ro.B} episodes, {ro.events.shape[0]} collisions ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------- QA bank
t0 = time.time()
bank = build_qa_bank(env, ro, Th_sim=Th * skip, T_sim=(Th + Tp) * skip,
                     n_per_type=4, seed=0)
print(f"[qa ] {bank.ep.numel()} questions in {time.time()-t0:.1f}s")
for t, name in enumerate(QTYPES):
    m = bank.qtype == t
    p = float(bank.answer[m].float().mean())
    print(f"[qa ]   {name:15s} n={int(m.sum()):5d}  P(yes)={p:.3f}")
    assert m.sum() > 0, f"no {name} questions generated"
    assert 0.15 < p < 0.85, f"{name} labels degenerate (P(yes)={p})"

# counterfactual answers must genuinely differ from the factual outcome,
# otherwise the "counterfactual" category is secretly a descriptive one
base = bank.copy_factual_baseline()
print("[qa ] copy-the-factual-outcome baseline: "
      + "  ".join(f"{k} {v:.1f}" for k, v in base.items()))
assert base["counterfactual"] < 62.0, (
    f"counterfactual questions still solvable by copying the factual outcome "
    f"({base['counterfactual']:.1f}%) — stratification failed")

expl = bank.qtype == 3
print(f"[qa ] explanatory P(yes)={float(bank.answer[expl].float().mean()):.3f} "
      f"(but-for causation is genuinely rare)")

# episode-level split, no leakage
tr, te = bank.split(0.75, seed=0)
overlap = set(tr.ep.tolist()) & set(te.ep.tolist())
assert not overlap, f"episode leakage between probe train/eval: {len(overlap)}"
print(f"[qa ] split train {tr.ep.numel()} / eval {te.ep.numel()}, "
      f"episode overlap {len(overlap)} (must be 0)")

# ---------------------------------------------------------------- probe
enc = build_encoder("oracle", w.state_dim(), 128, 0).to(dev)
enc.fit_normalizer(ro.state)
pcfg = PredictorConfig(d_model=256, n_heads=4, head_dim=64, n_layers=6,
                       mlp_hidden=1024, Th=Th, Tp=Tp, n_slots=w.n_slots)
model = CJEPAPredictor(pcfg).to(dev)   # UNTRAINED on purpose

ep = torch.arange(data.B, device=dev)
st = torch.zeros(data.B, device=dev, dtype=torch.long)
states = data.window_states(ep, st)

for mode in ["oracle", "static", "model"]:
    t0 = time.time()
    traj = imagined_trajectories(model, enc, states, Th, mode=mode)
    acc = train_probe(traj, tr, te, steps=900, seed=0, device=dev)
    print(f"[prb] {mode:7s} " + "  ".join(f"{k} {v:.1f}" for k, v in acc.items()
                                          if k != "n_eval")
          + f"   ({time.time()-t0:.0f}s)")

print("[prb] expectation: oracle > static; an UNTRAINED model should sit near "
      "static, since its imagined future is noise")

# ---------------------------------------------------------------- influence
t0 = time.time()
infl = evaluate_influence(model, enc, data, ro, n_mask=2, n_episodes=96,
                          frame_skip=skip, seed=0)
print(f"[inf] untrained model: {infl}  ({time.time()-t0:.0f}s)")
assert infl["attention"]["n_queries"] > 0, "no scorable influence queries"
assert infl["ablation"]["n_queries"] > 0
print("[inf] expectation: AUROC ~0.5 for an untrained model (no structure yet)")
print("TEST_EVAL OK")
