"""
Direct, probe-free counterfactual evaluation of the world model.

WHY THIS EXISTS ALONGSIDE THE ALOE PROBE
----------------------------------------
Tab. 1's metric is VQA accuracy through a trained probe. That is faithful to the
paper, but it measures the world model only through a second learned model, and a
probe that is too weak flattens every arm toward chance. We therefore add a
measurement that touches the world model directly and has no learned parts at all.

Because our oracle encoder is exactly invertible (`encoder.decode`), we can read
the model's imagined future back out as physical coordinates and compare it to a
re-simulation of the true physics. No probe, no training, no label noise.

THE QUERY
---------
This is a Pearl-style counterfactual, and the intervention point matters:

    factual past  ->  do(remove object k)  ->  future

  * The model is given the FACTUAL history of every other object, with object k's
    slot set to the empty-slot encoding across the whole history window. Empty
    slots are in-distribution (episodes contain 4-6 objects in 7 slots), so this
    is not an off-manifold input.
  * Ground truth branches the simulator from the last observed frame with k
    deleted, and rolls forward.

We branch at the end of the history rather than at t = 0 (which is what CLEVRER's
counterfactual videos do) because the model is *conditioned* on the factual past.
Asking "what if k had never existed" while handing it a past that k demonstrably
influenced would be an ill-posed query. Intervening at the observation boundary
is the well-posed version, and we state the choice rather than bury it.

THE SCORE
---------
For each (episode, deleted object k) we compute mean position error over the
surviving objects:

    d_model  = || model prediction under do(remove k)  -  true counterfactual ||
    d_ignore = || true FACTUAL future                  -  true counterfactual ||

`d_ignore` is the error of the degenerate strategy "ignore the intervention and
predict what actually happened". So

    cf_gain = 1 - d_model / d_ignore

      1.0  perfect counterfactual prediction
      0.0  the intervention was ignored entirely
     <0.0  worse than ignoring it

We restrict to cases where the deletion actually matters (`d_ignore` above a
threshold); otherwise the ratio is dominated by episodes where removing k changes
nothing and the metric measures nothing. We also report the model's ordinary
forward error on the unmodified history, so `cf_gain` can be read against how
good the model is at plain prediction.
"""

from __future__ import annotations

import torch

from ..envs.interaction_world import InteractionWorld


@torch.no_grad()
def counterfactual_rollout(
    model,
    encoder,
    env: InteractionWorld,
    states: torch.Tensor,      # (B, T, N, D) factual window, T = Th + Tp
    frame_skip: int = 2,
    min_effect: float = 0.02,  # world is [0,1]^2; ignore no-op deletions
):
    """Return counterfactual-prediction metrics for a batch of episodes."""
    model.eval()
    cfg = model.cfg
    Th, Tp = cfg.Th, cfg.Tp
    dev = states.device
    B, T, N, D = states.shape
    n_colors = env.cfg.n_colors

    present = states[:, 0, :, 5] > 0.5                       # (B,N)

    # ---- model's ordinary forward prediction (no intervention) -------------- #
    z = encoder(states)
    pred_fact = encoder.decode(model.rollout(z[:, :Th]).float())      # (B,Tp,N,D)
    gt_fact = states[:, Th:]                                          # (B,Tp,N,D)
    fwd_err = (pred_fact[..., :2] - gt_fact[..., :2]).pow(2).sum(-1).sqrt()

    valid_f = (gt_fact[..., 5] > 0.5)
    fwd_err_mean = float((fwd_err * valid_f).sum() / valid_f.sum().clamp_min(1))

    # ---- branch point: last observed frame ---------------------------------- #
    branch = states[:, Th - 1]                                        # (B,N,D)
    n_sim = frame_skip * Tp + 1
    take = torch.arange(1, Tp + 1, device=dev) * frame_skip           # sim offsets

    gains, d_models, d_ignores, ks = [], [], [], []

    for k in range(N):
        sel = present[:, k]
        if sel.sum() == 0:
            continue

        # ---- ground-truth counterfactual: delete k, re-simulate ------------- #
        init = InteractionWorld.unpack(branch, n_colors)
        dele = torch.full((B,), k, device=dev, dtype=torch.long)
        dele = torch.where(sel, dele, torch.full_like(dele, -1))
        ro_cf = env.counterfactual(init, dele, n_frames=n_sim)
        gt_cf = ro_cf.state[:, take]                                  # (B,Tp,N,D)

        # ---- model under do(remove k): zero k's slot across the history ----- #
        s_int = states.clone()
        s_int[:, :Th, k, :] = 0.0                                     # empty slot
        z_int = encoder(s_int)
        pred_cf = encoder.decode(model.rollout(z_int[:, :Th]).float())

        # score only the SURVIVING objects
        keep = present.clone()
        keep[:, k] = False
        m = keep[:, None, :].expand(B, Tp, N) & (gt_cf[..., 5] > 0.5)

        d_model = ((pred_cf[..., :2] - gt_cf[..., :2]).pow(2).sum(-1).sqrt() * m
                   ).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)
        d_ignore = ((gt_fact[..., :2] - gt_cf[..., :2]).pow(2).sum(-1).sqrt() * m
                    ).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)

        ok = sel & (d_ignore > min_effect)      # deletion must actually matter
        if ok.sum() == 0:
            continue
        d_models.append(d_model[ok])
        d_ignores.append(d_ignore[ok])
        gains.append(1.0 - d_model[ok] / d_ignore[ok])
        ks.append(torch.full((int(ok.sum()),), k, device=dev))

    if not gains:
        return {"cf_gain": float("nan"), "n": 0, "fwd_err": fwd_err_mean}

    gains = torch.cat(gains)
    d_models = torch.cat(d_models)
    d_ignores = torch.cat(d_ignores)
    return {
        "cf_gain": float(gains.mean()),
        "cf_gain_median": float(gains.median()),
        "d_model": float(d_models.mean()),
        "d_ignore": float(d_ignores.mean()),
        "frac_better_than_ignoring": float((gains > 0).float().mean()),
        "n": int(gains.numel()),
        "fwd_err": fwd_err_mean,
    }


@torch.no_grad()
def collision_prediction(model, encoder, states: torch.Tensor, rollout_gt=None):
    """Does the imagined future contain the right *collisions*?

    Position error averages over long ballistic stretches where prediction is
    trivial, so it can look fine while the model misses every interaction. Here
    we decode the imagined future, detect predicted collisions geometrically
    (centre distance < sum of radii) and score them against collisions detected
    the same way in the true future. This isolates exactly the interaction-
    dependent part of the dynamics — the part object-level masking is supposed
    to fix.
    """
    model.eval()
    cfg = model.cfg
    Th, Tp = cfg.Th, cfg.Tp
    dev = states.device
    B, T, N, _ = states.shape

    z = encoder(states)
    pred = encoder.decode(model.rollout(z[:, :Th]).float())     # (B,Tp,N,D)
    gt = states[:, Th:]

    def contacts(x):
        p = x[..., :2]                                          # (B,Tp,N,2)
        r = x[..., 4]
        pres = x[..., 5] > 0.5
        d = torch.cdist(p, p)                                   # (B,Tp,N,N)
        rad = r[..., :, None] + r[..., None, :]
        c = (d < rad * 1.05) & pres[..., :, None] & pres[..., None, :]
        idx = torch.arange(N, device=dev)
        c[..., idx, idx] = False
        return c.any(dim=1)                                     # (B,N,N) over horizon

    # radii/presence come from ground truth: we are testing whether the model got
    # the *motion* right, not whether the decoder recovered a constant attribute
    pred_full = pred.clone()
    pred_full[..., 4:6] = gt[..., 4:6]

    cp, cg = contacts(pred_full), contacts(gt)
    iu, ju = torch.triu_indices(N, N, offset=1, device=dev)
    P, G = cp[:, iu, ju], cg[:, iu, ju]

    tp = float((P & G).sum())
    fp = float((P & ~G).sum())
    fn = float((~P & G).sum())
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, tp + fn)
    return {
        "collision_f1": 2 * prec * rec / max(1e-9, prec + rec),
        "collision_precision": prec,
        "collision_recall": rec,
        "n_true_collisions": int(G.sum()),
    }
