"""
The C-JEPA training objective (Eq. 5 and Eq. 6).

Eq. 5 -- masked latent prediction over the full history+future interval Tbar:

    L_mask = E[ sum_{tau in Tbar} sum_{i=1..N}
                1[ zbar^i_tau != z^i_tau ] * || zhat^i_tau - z^i_tau ||_2^2 ]

The indicator "selects all tokens masked in the input over Tbar". Note what is
*absent*: there is no reconstruction loss, no decoder, no pixel term. Tab. 2 is
built around this — SlotFormer loses 36 points of counterfactual accuracy when
its reconstruction loss is removed, whereas C-JEPA never had one.

Eq. 6 -- the same quantity split into the two halves that do different jobs:

    L_mask = L_history + L_future

    "The history term suppresses reliance on trivial self-dynamics under partial
     observability, while the future term enforces alignment with forward world
     modeling."

OC-JEPA is exactly the |M| = 0 case: L_history vanishes (nothing in the history
is masked) and only L_future survives. So the OC-JEPA -> C-JEPA delta in Tab. 1
*is* the contribution of L_history. We therefore always log the two terms
separately — it is the single most diagnostic number in the whole reproduction.

NORMALISATION NOTE
------------------
Eq. 5 sums squared errors over tokens. We optimise the *mean over masked tokens*
so that the effective learning rate does not scale with |M| — otherwise a run at
|M| = 4 would take ~4x larger steps than |M| = 1 and the sweep would confound
masking budget with optimisation strength. `reduction="sum_per_sample"` recovers
the literal Eq. 5 for anyone who wants it.
"""

from __future__ import annotations

import torch


def masked_latent_loss(
    zhat: torch.Tensor,        # (B, T, N, d)
    z: torch.Tensor,           # (B, T, N, d)
    mask: torch.Tensor,        # (B, T, N) bool
    Th: int,
    reduction: str = "mean",
    valid: torch.Tensor | None = None,   # (B, T, N) bool — exclude empty slots
):
    """Return (total, history_term, future_term, n_masked)."""
    m = mask
    if valid is not None:
        m = m & valid

    se = (zhat - z).pow(2).sum(-1)          # (B, T, N)  squared L2 per token
    sel = se * m.float()

    hist_m = m.clone()
    hist_m[:, Th:] = False
    fut_m = m.clone()
    fut_m[:, :Th] = False

    n_all = m.sum().clamp_min(1)
    n_h = hist_m.sum().clamp_min(1)
    n_f = fut_m.sum().clamp_min(1)

    if reduction == "mean":
        total = sel.sum() / n_all
    elif reduction == "sum_per_sample":
        total = sel.sum() / zhat.shape[0]
    else:
        raise ValueError(reduction)

    l_hist = (se * hist_m.float()).sum() / n_h
    l_fut = (se * fut_m.float()).sum() / n_f
    return total, l_hist.detach(), l_fut.detach(), int(m.sum())


def slot_validity(states: torch.Tensor) -> torch.Tensor:
    """(B, T, N, D) -> (B, T, N) bool: True where the slot holds a real object.

    Empty slots encode to a constant (all-zero state -> bias b), so including
    them in the loss would let the model bank free error reduction on tokens
    that carry no dynamics. Excluding them keeps the reported MSE comparable
    across episodes with different object counts.
    """
    return states[..., 5] > 0.5
