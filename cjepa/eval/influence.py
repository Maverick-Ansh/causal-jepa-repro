"""
Validating the *influence neighborhood* (Def. 1, Thm. 1, Cor. 1).

WHY THIS FILE EXISTS
--------------------
This is the one place where we go beyond the paper rather than reproduce it.
The authors' own limitation section says:

    "while we formally characterize influence neighborhoods, we do not directly
     validate them on datasets with explicit temporal causal graphs, leaving this
     to future work."  (Sec. 7)

They fall back on a *qualitative* attention study on PHYRE (App. J), because
neither CLEVRER nor PHYRE ships a ground-truth temporal interaction graph.

Our simulator does. Every object-object collision is logged with its timestep, so
for a masked object i we know exactly which other objects carried information
about it. That turns Cor. 1 into a measurable claim:

    Cor. 1: "Optimizing L_mask under repeated exposure to diverse object-level
             masking encourages state-dependent attention patterns that align with
             the influence neighborhood N_t(i)."

    => C-JEPA (|M| > 0) should align with the true interaction graph better than
       OC-JEPA (|M| = 0), and alignment should track |M|.

TWO MEASUREMENTS, ONE WEAK AND ONE STRONG
-----------------------------------------
1. `attention_influence` — the paper's own proxy. Following SPARTAN (Lei et al.,
   2025) and App. J, read cross-slot attention from the masked queries of object
   i to the tokens of object j. Cheap, but attention weight is famously not the
   same thing as functional dependence.

2. `ablation_influence` — a *causal* probe of the model's computation, which the
   paper does not do. Ablate object j from the predictor's context and measure
   how much worse the prediction of masked object i becomes:

       influence(i <- j) = || zhat_i(context) - z_i ||^2  under j ablated
                         - || zhat_i(context) - z_i ||^2  under full context

   This is much closer to what Def. 1 actually asserts — that N_t(i) is the
   *minimal sufficient subset*, i.e. that dropping a member must hurt. A model
   can attend to j without using it; it cannot be hurt by losing j without using
   it.

Both are scored the same way: rank all candidate objects j for a given (episode,
masked object i) and ask how well that ranking recovers i's true collision
partners (AUROC + precision@1).
"""

from __future__ import annotations

import torch

from ..models.masking import object_mask


# --------------------------------------------------------------------------- #
# Ranking metric
# --------------------------------------------------------------------------- #
def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC over a single ranking (scores/labels are 1-D, labels in {0,1})."""
    pos, neg = labels > 0.5, labels < 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    sp, sn = scores[pos], scores[neg]
    # P(score_pos > score_neg) with ties counted as 0.5
    cmp = (sp[:, None] > sn[None, :]).float() + 0.5 * (sp[:, None] == sn[None, :]).float()
    return float(cmp.mean())


def score_alignment(scores: torch.Tensor, gt: torch.Tensor,
                    present: torch.Tensor, masked: torch.Tensor):
    """Aggregate ranking quality over all (episode, masked object) pairs.

    scores  : (B, N, N)  scores[b, i, j] = how much i depends on j (j != i)
    gt      : (B, N, N)  1.0 where i and j truly collided in the window
    present : (B, N)     real objects
    masked  : (B, N)     which objects were masked (only those are queries)
    """
    B, N, _ = scores.shape
    aurocs, p_at_1, n_q = [], [], 0
    for b in range(B):
        for i in range(N):
            if present[b, i] < 0.5 or masked[b, i] < 0.5:
                continue
            cand = present[b].clone()
            cand[i] = 0.0                      # never rank an object against itself
            idx = cand.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            lab = gt[b, i, idx]
            if lab.sum() == 0 or lab.sum() == idx.numel():
                continue                       # degenerate ranking, skip
            s = scores[b, i, idx]
            a = _auroc(s, lab)
            if a == a:                          # not NaN
                aurocs.append(a)
                p_at_1.append(float(lab[s.argmax()]))
                n_q += 1
    if not aurocs:
        return {"auroc": float("nan"), "p_at_1": float("nan"), "n_queries": 0}
    return {
        "auroc": float(torch.tensor(aurocs).mean()),
        "p_at_1": float(torch.tensor(p_at_1).mean()),
        "n_queries": n_q,
    }


# --------------------------------------------------------------------------- #
# 1. Attention proxy (the paper's App. J measurement, made quantitative)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_influence(model, encoder, states: torch.Tensor, n_mask: int = 2,
                        seed: int = 0, layers: str = "mean"):
    """Cross-slot attention from masked queries of i to tokens of j.

    Returns (scores (B,N,N), masked (B,N)).
    """
    model.eval()
    cfg = model.cfg
    dev = states.device
    B, T, N, _ = states.shape
    Th = cfg.Th

    z = encoder(states)
    g = torch.Generator(device=dev).manual_seed(seed)
    mask = object_mask(B, N, Th, cfg.Tp, n_mask, dev, count_mode="fixed", generator=g)
    _, attns = model(z, mask, need_attn=True)

    A = torch.stack(attns, 0)                       # (L, B, H, Lq, Lk)
    A = A.mean(2)                                    # average heads
    A = A.mean(0) if layers == "mean" else A[-1]     # (B, Lq, Lk)

    # query = object i at masked history steps (t0 is the visible anchor)
    q_t = torch.arange(1, Th, device=dev)
    k_t = torch.arange(0, Th, device=dev)
    qi = (q_t[:, None] * N + torch.arange(N, device=dev)[None, :]).reshape(-1)   # (Tq*N)
    ki = (k_t[:, None] * N + torch.arange(N, device=dev)[None, :]).reshape(-1)   # (Tk*N)

    sub = A[:, qi][:, :, ki]                         # (B, Tq*N, Tk*N)
    sub = sub.reshape(B, Th - 1, N, Th, N)
    scores = sub.mean(dim=(1, 3))                    # (B, N, N) average over time
    masked = mask[:, 1:Th].any(1).float()            # (B, N)
    return scores, masked


# --------------------------------------------------------------------------- #
# 2. Causal ablation probe (our addition)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ablation_influence(model, encoder, states: torch.Tensor, n_mask: int = 2,
                       seed: int = 0):
    """influence(i <- j) = rise in prediction error for masked i when j is ablated.

    "Ablated" means j's *own* tokens are replaced by its mask token everywhere
    except t0 — i.e. we apply the same observability intervention to j that
    training applies to masked objects (Lemma 1). We deliberately do not zero the
    token out, because a never-seen all-zero input would confound "j was removed"
    with "the input is off-distribution".
    """
    model.eval()
    cfg = model.cfg
    dev = states.device
    B, T, N, _ = states.shape
    Th, Tp = cfg.Th, cfg.Tp

    z = encoder(states)
    g = torch.Generator(device=dev).manual_seed(seed)
    base_mask = object_mask(B, N, Th, Tp, n_mask, dev, count_mode="fixed", generator=g)
    masked = base_mask[:, 1:Th].any(1).float()                # (B, N)

    valid = states[..., 5] > 0.5                              # (B,T,N)

    def err_per_object(mask):
        zhat, _ = model(z, mask)
        se = (zhat - z).pow(2).sum(-1)                        # (B,T,N)
        sel = base_mask & valid                               # score only the
        sel = sel.clone(); sel[:, Th:] = False                # masked history of i
        num = (se * sel.float()).sum(1)                       # (B,N)
        den = sel.float().sum(1).clamp_min(1e-6)
        return num / den

    base_err = err_per_object(base_mask)                      # (B,N)

    scores = torch.zeros(B, N, N, device=dev)
    for j in range(N):
        m = base_mask.clone()
        m[:, 1:Th, j] = True                                  # ablate j's history
        e = err_per_object(m)                                 # (B,N)
        scores[:, :, j] = e - base_err                        # rise in error for each i
    # self-influence is meaningless (i was already masked)
    idx = torch.arange(N, device=dev)
    scores[:, idx, idx] = -1e9
    return scores, masked


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_influence(model, encoder, data, rollout, n_mask: int = 2,
                       n_episodes: int = 256, window_start: int = 0,
                       frame_skip: int = 2, seed: int = 0):
    """Run both probes on a held-out slice and score them against ground truth."""
    dev = data.states.device
    B = min(n_episodes, data.B)
    ep = torch.arange(B, device=dev)
    st = torch.full((B,), window_start, device=dev, dtype=torch.long)
    states = data.window_states(ep, st)                        # (B,T,N,D)

    Th, Tp = model.cfg.Th, model.cfg.Tp
    # sim-frame window covered by this latent window
    lo = window_start * frame_skip
    hi = (window_start + Th + Tp) * frame_skip
    gt = rollout.interaction_graph(lo, hi)[:B]                 # (B,N,N)
    present = rollout.present[:B]

    out = {}
    s_att, msk = attention_influence(model, encoder, states, n_mask=n_mask, seed=seed)
    out["attention"] = score_alignment(s_att, gt, present, msk)

    s_abl, msk2 = ablation_influence(model, encoder, states, n_mask=n_mask, seed=seed)
    out["ablation"] = score_alignment(s_abl, gt, present, msk2)
    return out
