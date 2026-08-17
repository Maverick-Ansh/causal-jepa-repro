"""Generate notebooks/causal_jepa_repro.ipynb from source.

The notebook is a build artifact — edit THIS file, then:
    python -m scripts.build_notebook
Keeping the notebook generated avoids the usual problem of prose and code
drifting apart inside a hand-edited .ipynb with stale outputs baked in.
"""

from __future__ import annotations

import json
import os

REPO = "https://github.com/Maverick-Ansh/causal-jepa-repro"

CELLS: list[tuple[str, str]] = []


def md(s: str):
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str):
    CELLS.append(("code", s.strip("\n")))


# --------------------------------------------------------------------------- #
md(f"""
# Causal-JEPA, reproduced and annotated

An independent reproduction of **Causal-JEPA: Learning World Models through
Object-Level Latent Masking** ([arXiv:2602.11389](https://arxiv.org/abs/2602.11389)
— Nam, Le Lidec, Maes, LeCun, Balestriero, ICML 2026).

Repo: [{REPO}]({REPO}) · full write-up in `REPORT.md`

---

### The idea in one paragraph

A world model built on object slots can cheat. Between collisions an object moves
in a straight line, so a predictor can score well by extrapolating each object
from its own past and never modelling interactions at all. C-JEPA removes the
cheat by construction: during training it masks a whole object's history — every
timestep except the earliest one, which is kept as an *identity anchor* — so that
object's state can only be recovered by looking at what the other objects did.
The loss is a plain masked L2 in latent space (no decoder, no pixel
reconstruction), and the claim is that this single change makes interaction
reasoning *necessary* rather than merely possible.

### What this notebook does

Runs the reproduction end to end on a small GPU. The heavy sweep lives in
`scripts/run_sweep.py`; here we walk through the pieces and check that each one
behaves the way the paper says it should.
""")

code(f"""
!git clone -q {REPO} 2>/dev/null || (cd causal-jepa-repro && git pull -q)
%cd causal-jepa-repro
import torch
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
""")

md("""
## 1. The world

CLEVRER is "collision events for video representation and reasoning" — objects
sliding on a plane and bouncing off each other. We rebuild that dynamical
structure directly: elastic collisions in a box, deterministic, fully vectorised
on the GPU.

Two things this buys us that CLEVRER cannot:

* **exact counterfactuals** — delete an object, re-run the same physics
* **an exact temporal interaction graph** — every collision logged as `(t, i, j)`

The second one matters a lot later: the paper's own limitations section says the
influence neighborhoods it defines were never validated against a ground-truth
interaction graph. With this world, they can be.

Object count varies (4–6) inside a fixed 7 slots, mirroring App. C.1 — CLEVRER
has at most 6 visible objects and the paper uses 7 slots, "where one slot
implicitly captures background or empty regions".
""")

code("""
from cjepa.envs import InteractionWorld, WorldConfig
import torch

cfg = WorldConfig(n_frames=64)
env = InteractionWorld(cfg, device='cuda')
ro  = env.generate(512, seed=0)

print('states', tuple(ro.state.shape), '= (episodes, frames, slots, state_dim)')
print(f'object-object collisions: {ro.events.shape[0]} '
      f'({ro.events.shape[0]/ro.B:.2f} per episode)')

# elastic collisions must conserve kinetic energy exactly - a physics bug here
# would quietly become a "hard to predict" dataset and confound everything
m  = ro.state[..., 4] ** 2
ke = (0.5 * m * (ro.state[..., 2:4] ** 2).sum(-1)).sum(-1)
print(f'mean |KE drift| across an episode: '
      f'{((ke[:, -1] - ke[:, 0]).abs() / ke[:, 0]).mean():.6%}')
""")

code("""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='white')
for ax, b in zip(axes, [0, 1, 2]):
    n = int(ro.present[b].sum())
    for i in range(n):
        p = ro.state[b, :, i, :2].cpu()
        ax.plot(p[:, 0], p[:, 1], linewidth=1.6, alpha=0.9)
        ax.scatter(*p[0], s=28, zorder=3)
    hits = ro.events[(ro.events[:, 0] == b)]
    for e in hits:
        t, i = int(e[1]), int(e[2])
        ax.scatter(*ro.state[b, t, i, :2].cpu(), marker='x', c='k', s=44, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(f'episode {b} — {n} objects, {len(hits)} collisions', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Trajectories (x = object-object collision)', x=0.02, ha='left')
fig.tight_layout(); plt.show()
""")

md("""
## 2. The masking rule (Sec. 4.2, Eq. 3)

This is the whole contribution, so it is worth checking against the text
literally rather than trusting the implementation.

> "While future entity tokens are always masked for prediction, we additionally
> mask the observable latent states of selected objects across the history window
> T according to a masking index set M, **preserving only the earliest time step
> t0 as an identity anchor**."

and the masked token itself:

$$\\tilde{z}^i_\\tau = \\phi(z^i_{t_0}) + e_\\tau$$

The anchor is load-bearing. The predictor has **no positional encoding over the
slot dimension** (slots are permutation-equivariant, so indexing them would break
the symmetry) — which means `φ(z^i_{t0})` is the *only* thing telling the model
which object a masked token belongs to. We ablate the anchor later; it should be
catastrophic, not merely worse.
""")

code("""
from cjepa.models.masking import object_mask
import matplotlib.pyplot as plt

Th, Tp, N = 6, 10, 7
M = object_mask(1, N, Th, Tp, n_mask=3, device='cuda', count_mode='fixed')

assert M[:, Th:, :].all(),      'every future token must be masked'
assert not M[:, 0, :].any(),    't0 anchor must stay visible for ALL objects'
assert (M[:, 1:Th, :].any(1).sum(-1) == 3).all(), '|M| must be honoured'
print('masking law matches Sec. 4.2')

fig, ax = plt.subplots(figsize=(7, 3.2), facecolor='white')
ax.imshow(M[0].cpu().T, aspect='auto', cmap='Blues', vmin=0, vmax=1.6)
ax.axvline(Th - 0.5, color='#eb6834', linewidth=2)
ax.set_xlabel('time  ->'); ax.set_ylabel('object slot')
ax.set_title('masked (dark) vs visible (light).  orange = history | future boundary\\n'
             'column 0 stays visible for every slot: that is the identity anchor',
             fontsize=10, loc='left')
ax.set_xticks(range(Th + Tp)); ax.set_yticks(range(N))
fig.tight_layout(); plt.show()
""")

md("""
## 3. Encoder, predictor, loss

The encoder is **frozen** — the paper never trains it (Sec. 3; App. E.3 works "on
pre-extracted object embeddings"), so it sits outside the contribution and we can
substitute a cheap one. We use two, because Tab. 1 contains a result that is easy
to miss: *the best masking budget depends on encoder quality*. With VideoSAUR,
accuracy climbs all the way to |M| = 4; with the weaker SAVi it peaks at |M| = 2
and then collapses. So we need a strong encoder and a deliberately degraded one.

The degraded encoder models how slot attention actually fails — it **bleeds**
content between slots, so a "slot" is a blend of objects rather than one object.

The predictor is a bidirectional ViT-style transformer (App. E.2: 6 layers, head
dim 64, MLP 2048) and the loss is Eq. 5 — masked L2 in latent space, nothing else.
Eq. 6 splits it into a history term and a future term, and that split is the most
diagnostic number in the reproduction: **OC-JEPA is exactly the |M| = 0 case, where
the history term vanishes.** So the OC-JEPA → C-JEPA delta *is* the history term.
""")

code("""
from cjepa.encoders import build_encoder
from cjepa.models import CJEPAPredictor, PredictorConfig, build_mask
from cjepa.models.losses import masked_latent_loss, slot_validity
from cjepa.data import SlotSequenceData

enc = build_encoder('oracle', cfg.state_dim(), 128, seed=0).cuda().fit_normalizer(ro.state)
data = SlotSequenceData(ro, Th, Tp, frame_skip=2, device='cuda')   # App. E.3: stride 2

pcfg  = PredictorConfig(d_model=256, n_heads=4, head_dim=64, n_layers=6,
                        mlp_hidden=1024, Th=Th, Tp=Tp, n_slots=N)
model = CJEPAPredictor(pcfg).cuda()
print(f'predictor: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

s, _, _ = data.sample_batch(64)
z = enc(s)
for n_mask in [0, 2, 4]:
    mk = build_mask('object', 64, N, Th, Tp, n_mask, 'cuda')
    zh, _ = model(z, mk)
    tot, lh, lf, _ = masked_latent_loss(zh, z, mk, Th, valid=slot_validity(s))
    print(f'|M|={n_mask}:  L_total {tot.item():7.2f}   '
          f'L_history {lh.item():7.2f}   L_future {lf.item():7.2f}'
          + ('   <- OC-JEPA: no history term at all' if n_mask == 0 else ''))
""")

md("""
## 4. Does masking change what the model learns?

A quick side-by-side: same architecture, same data, same steps — only `|M|`
differs. `L_future` is what both models are ultimately judged on at inference
time, since inference masks *only* the future (Sec. 4.2, "Inference").

This is a short demo run. The real numbers come from `scripts/run_sweep.py`
(2 encoders × 5 masking budgets × 2 seeds) and land in `REPORT.md`.
""")

code("""
from cjepa.train import TrainConfig, train_cjepa

demo = {}
for n_mask in [0, 3]:
    tcfg = TrainConfig(n_mask=n_mask, epochs=4, batch_size=256, lr=5e-4,
                       seed=42, tag=f'demo_M{n_mask}')
    _, res = train_cjepa(PredictorConfig(d_model=256, n_heads=4, head_dim=64,
                                         n_layers=6, mlp_hidden=1024,
                                         Th=Th, Tp=Tp, n_slots=N),
                         tcfg, enc, data, data, device='cuda')
    demo[n_mask] = res['final']
    print()

print(f"{'':12s} {'val latent MSE':>15s} {'position error':>16s}")
for k, v in demo.items():
    name = 'OC-JEPA' if k == 0 else f'C-JEPA |M|={k}'
    print(f'{name:12s} {v["val_latent_mse"]:15.3f} {v["val_pos_err"]:16.4f}')
""")

md("""
## 5. Influence neighborhoods — going past the paper

Definition 1 defines the **influence neighborhood** `N_t(i)`: the minimal set of
other variables sufficient to predict masked object *i*. Theorem 1 says a
predictor that ignores `N_t(i)` cannot reach minimal loss, and Corollary 1 says
training under object masking should make attention align with it.

The paper can only check this *qualitatively*, on PHYRE, via attention maps —
because no benchmark it uses ships a ground-truth interaction graph. Ours does.
So we score two probes against the true collision graph:

1. **attention** — the paper's own proxy (App. J), made quantitative
2. **ablation** — a *causal* probe: ablate object *j* from the context and measure
   how much worse the prediction of masked *i* gets. A model can attend to *j*
   without using it; it cannot be *hurt* by losing *j* without using it.
""")

code("""
from cjepa.eval.influence import evaluate_influence
print(evaluate_influence(model, enc, data, ro, n_mask=2, n_episodes=128, frame_skip=2))
print('\\nAUROC 0.5 = attention/dependence is unrelated to who actually collided.')
print('Trained-model numbers across the full |M| sweep are in REPORT.md (Fig. 2).')
""")

md("""
## 6. Results

The full sweep, tables, and figures are produced by:

```bash
python -m scripts.run_sweep --shard 0 --n_shards 2   # one process per GPU
python -m scripts.eval_checkpoints                   # probe-free metrics
python -m scripts.planning_cost                      # Tab. 3 efficiency claim
python -m scripts.aggregate                          # tables + figures
```

**Read `REPORT.md` for the findings** — including which claims reproduced, which
did not, and the several places where getting the measurement right turned out to
be harder than getting the model right.
""")

# --------------------------------------------------------------------------- #
def build():
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": (src + "\n").splitlines(True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join("notebooks", "causal_jepa_repro.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
