"""
Object-level latent masking — the entire contribution of the paper (Sec. 4.2).

WHAT THE PAPER SAYS, LINE BY LINE
--------------------------------
1.  "While future entity tokens are always masked for prediction, we additionally
     mask the observable latent states of selected objects across the history
     window T according to a masking index set M, preserving only the earliest
     time step t0 as an identity anchor."

    => for i in M:   history tokens (t0, t] are masked;  token at t0 stays visible
       for all i:    every future token (t, t + Tp] is masked

2.  "The identity anchor is introduced to address the permutation equivariance of
     slot-based representations with respect to object ordering, which motivates
     omitting positional encodings along the entity dimension. As a result,
     identity information of each entity must be provided for masked tokens,
     enabling the transformer predictor to distinguish *which* entities are
     masked."

    => there is NO positional encoding over slots. The only thing that tells the
       predictor "this masked token is object 3" is the anchor content itself.
       This is why removing the anchor breaks the method (we ablate it).

3.  Eq. 3:   z~^i_tau  =  phi(z^i_t0)  +  e_tau
    where phi is a linear projection and e_tau is "a learnable embedding combined
    with temporal positional encoding".

4.  Auxiliaries (actions / proprioception) are drawn in Fig. 1 as
    ": Observable auxiliaries" — they are never masked.

AN AMBIGUITY WE HAD TO RESOLVE
------------------------------
Fig. 1 says  `M ~ Uniform({1, ..., N})`  (the *indices* are uniform, |M| fixed),
but App. E.3 says "object-level masking is applied by randomly masking **between
zero and four** object slots for CLEVRER" (|M| itself is random).

Those are different training distributions, and Tab. 1 reports one number per
"|M|", so we support both and default to the appendix reading:

    count_mode="uniform_upto"  ->  k ~ U{0..|M|}, then choose k indices  (App. E.3)
    count_mode="fixed"         ->  k = |M| always                        (Fig. 1)

`scripts/run_sweep.py --count_mode both` runs the sweep under each so the report
can state which reading reproduces Tab. 1. This is exactly the sort of thing a
replication exists to pin down.

App. K additionally compares object-level masking against token- and tube-level
masking under a matched budget; all three live here.
"""

from __future__ import annotations

import torch


def object_mask(
    B: int,
    N: int,
    Th: int,
    Tp: int,
    n_mask: int,
    device,
    count_mode: str = "uniform_upto",
    anchor: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a bool tensor (B, Th+Tp, N): True == this token is masked.

    Args:
        n_mask:     |M|, the object masking budget.
        count_mode: "uniform_upto" (App. E.3) or "fixed" (Fig. 1).
        anchor:     keep t0 visible for masked objects. Setting False ablates the
                    identity anchor, which should be catastrophic — with no slot
                    positional encoding the predictor cannot tell masked objects
                    apart at all.
    """
    T = Th + Tp
    m = torch.zeros(B, T, N, dtype=torch.bool, device=device)

    # ---- future is always masked (Sec. 4.2) --------------------------------- #
    m[:, Th:, :] = True

    if n_mask > 0:
        if count_mode == "fixed":
            k = torch.full((B,), n_mask, device=device, dtype=torch.long)
        elif count_mode == "uniform_upto":
            k = torch.randint(0, n_mask + 1, (B,), device=device, generator=generator)
        else:
            raise ValueError(count_mode)

        # choose which objects to mask, uniformly over slots
        r = torch.rand(B, N, device=device, generator=generator)
        order = r.argsort(dim=-1)
        rank = order.argsort(dim=-1)                 # rank[b, i] in [0, N)
        chosen = rank < k[:, None]                   # (B, N) bool

        # mask the whole history for chosen objects ...
        m[:, :Th, :] |= chosen[:, None, :]
        # ... except t0, the identity anchor
        if anchor:
            m[:, 0, :] = False

    return m


def token_mask(
    B: int, N: int, Th: int, Tp: int, ratio: float, device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """App. K token-level masking: an i.i.d. random subset of history tokens."""
    T = Th + Tp
    m = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    m[:, Th:, :] = True
    if ratio > 0 and Th > 1:
        r = torch.rand(B, Th - 1, N, device=device, generator=generator)
        m[:, 1:Th, :] = r < ratio          # t0 never masked (parity with anchor)
    return m


def tube_mask(
    B: int, N: int, Th: int, Tp: int, n_tubes: int, tube_len: int, device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """App. K tube-level masking: contiguous (time x slot) blocks.

    "for tube masking we further set the number of masked tubes to correspond to
     the size of the object masking index set."
    """
    T = Th + Tp
    m = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    m[:, Th:, :] = True
    if n_tubes <= 0 or Th <= 1:
        return m
    L = max(1, min(tube_len, Th - 1))
    for _ in range(n_tubes):
        slot = torch.randint(0, N, (B,), device=device, generator=generator)
        start = torch.randint(1, max(2, Th - L + 1), (B,), device=device, generator=generator)
        offs = torch.arange(L, device=device)
        tt = (start[:, None] + offs[None, :]).clamp(max=Th - 1)      # (B, L)
        bb = torch.arange(B, device=device)[:, None].expand_as(tt)
        ss = slot[:, None].expand_as(tt)
        m[bb.reshape(-1), tt.reshape(-1), ss.reshape(-1)] = True
    return m


def build_mask(strategy: str, B, N, Th, Tp, budget, device, **kw) -> torch.Tensor:
    """Dispatch with a *budget-matched* interface (App. K)."""
    if strategy == "object":
        return object_mask(B, N, Th, Tp, int(budget), device,
                           count_mode=kw.get("count_mode", "uniform_upto"),
                           anchor=kw.get("anchor", True),
                           generator=kw.get("generator"))
    if strategy == "token":
        # match the *expected* fraction of history tokens an object mask hides
        ratio = float(budget) / N
        return token_mask(B, N, Th, Tp, ratio, device, generator=kw.get("generator"))
    if strategy == "tube":
        return tube_mask(B, N, Th, Tp, int(budget), kw.get("tube_len", Th - 1), device,
                         generator=kw.get("generator"))
    if strategy == "none":
        m = torch.zeros(B, Th + Tp, N, dtype=torch.bool, device=device)
        m[:, Th:, :] = True
        return m
    raise ValueError(strategy)
