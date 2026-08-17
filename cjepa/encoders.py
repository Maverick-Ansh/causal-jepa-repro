"""
Frozen object-centric encoders  g : X_t -> S_t = {s^1_t, ..., s^N_t},  s^i_t in R^d
(Eq. 1 of the paper).

WHY FROZEN, AND WHY WE ARE ALLOWED TO SWAP THE ENCODER
------------------------------------------------------
C-JEPA never trains the encoder. Sec. 3 ("Preliminaries"):

    "Following prior works, we employ frozen target encoders and optimize the
     predictor with a joint embedding prediction objective."

and App. E.3: "All experiments are conducted on a single GPU, on pre-extracted
object embeddings."

So the encoder is a *fixed feature map* sitting outside the contribution. The
paper's own headline ablation — OC-JEPA (|M| = 0) vs C-JEPA (|M| > 0), Tab. 1 —
holds the encoder fixed and varies only the masking objective. That is precisely
the comparison we reproduce, so substituting a cheaper frozen encoder changes the
absolute numbers but not the quantity under test.

THE TWO ENCODERS, AND WHY THERE ARE TWO
---------------------------------------
Tab. 1 has a result that is easy to miss and is the most interesting thing in the
paper: the *optimal masking budget depends on encoder quality*.

    VideoSAUR (strong encoder):  |M| = 0 -> 4  improves monotonically  (+21.13 CF)
    SAVi      (weak encoder):    peaks at |M| = 2, then *collapses* at |M| = 4 (-7.04)

The paper's explanation (Sec. 5.1): "excessive masking can remove informative
dependencies, indicating an optimal masking regime that depends on the robustness
of the underlying object representations from the encoder."

To reproduce that interaction we need two frozen encoders of *different quality*:

  * `OracleSlotEncoder`   — perfectly object-aligned. Satisfies Assumption 3
                            ("each slot corresponds to a coherent object-level
                            state variable") exactly. Stands in for VideoSAUR.
  * `DegradedSlotEncoder` — same map, but each slot is contaminated with a
                            fraction of the other slots' content before
                            projection. This is a faithful model of how SAVi
                            actually fails: slot attention *bleeds* mass between
                            objects, so a "slot" is a blend rather than one
                            object. Stands in for SAVi.

Both are deterministic and frozen — essential, because in a JEPA the encoder
output *is* the regression target. A stochastic encoder would make the target
noisy and the loss floor non-zero for reasons unrelated to the objective.

Slot dimensionality is 128 throughout, matching App. E.2:
"We maintain object latent states with slot dimensionality 128 throughout."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OracleSlotEncoder(nn.Module):
    """Frozen random linear projection of ground-truth object state -> R^d.

    Playing the role of VideoSAUR: a strong, object-aligned frozen encoder.

    We keep an explicit pseudo-inverse `decode()` so that *analysis* code can map
    predicted latents back to physical coordinates (to detect predicted
    collisions, measure position error, etc.). The predictor never sees it and it
    is never trained — it exists only so that we can read out what the world
    model imagined.
    """

    def __init__(self, state_dim: int, d: int = 128, seed: int = 0, scale: float = 1.0):
        super().__init__()
        self.state_dim, self.d = state_dim, d
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(state_dim, d, generator=g) / (state_dim ** 0.5) * scale
        b = torch.zeros(d)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        # Moore-Penrose pseudo-inverse for read-out (analysis only).
        self.register_buffer("W_pinv", torch.linalg.pinv(W))
        # Per-channel standardisation, fitted once on train data then frozen.
        # WITHOUT THIS THE REPLICATION SILENTLY FAILS: raw positions are O(1)
        # while velocities are O(0.02), so an isotropic random projection buries
        # velocity ~50x below position. The predictor would then be scored almost
        # entirely on position and could ignore dynamics. Real slot encoders
        # (VideoSAUR / SAVi) emit roughly unit-scale features, so standardising
        # here restores parity rather than adding information.
        self.register_buffer("mu", torch.zeros(state_dim))
        self.register_buffer("sd", torch.ones(state_dim))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def fit_normalizer(self, states: torch.Tensor) -> "OracleSlotEncoder":
        """Fit mu/sd over *present* objects only, then freeze."""
        flat = states.reshape(-1, self.state_dim)
        pres = flat[:, 5] > 0
        if pres.sum() > 1:
            flat = flat[pres]
        self.mu.copy_(flat.mean(0))
        self.sd.copy_(flat.std(0).clamp_min(1e-3))
        # `present` is a flag, not a measurement: keep it interpretable at 0/1
        self.mu[5], self.sd[5] = 0.0, 1.0
        return self

    @torch.no_grad()
    def _pre(self, state: torch.Tensor) -> torch.Tensor:
        out = (state - self.mu) / self.sd
        return out * state[..., 5:6]        # empty slots stay identically zero

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(..., N, state_dim) -> (..., N, d)"""
        return self._pre(state) @ self.W + self.b

    @torch.no_grad()
    def fit_readout(self, states: torch.Tensor, batch: int = 200_000) -> float:
        """Least-squares read-out  z -> state, fitted once on train data, frozen.

        WHY THIS IS NEEDED
        ------------------
        `decode` below inverts the projection W analytically, which is exact for
        the oracle encoder. It is WRONG for `DegradedSlotEncoder`: that encoder
        bleeds content between slots and pushes through a rank-limited projector,
        so its map is genuinely non-invertible — information is destroyed on
        purpose. Analytically "inverting" only W would silently produce garbage
        positions for the degraded arm, corrupting every position-space metric.

        The honest replacement is the best *linear estimate* of an object's state
        from its slot latent, fitted on training data. For the oracle encoder it
        recovers the exact inverse (residual ~0); for the degraded encoder it
        leaves a non-zero residual, which is real irreducible loss and is
        returned here so it can be reported as the floor for any position-space
        error measured through it.
        """
        # encode as a SET so slot-bleeding is applied exactly as at train time,
        # then pair each latent with its own object's state
        flat_s = states.reshape(-1, self.state_dim)
        flat_z = self.forward(states).reshape(-1, self.d)
        idx = (flat_s[:, 5] > 0.5).nonzero(as_tuple=True)[0]
        if idx.numel() > batch:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:batch]]
        flat_s, z = flat_s[idx], flat_z[idx]
        X = torch.cat([z, torch.ones_like(z[:, :1])], 1).double()
        Y = flat_s.double()
        # RIDGE, not lstsq. torch.linalg.lstsq on CUDA uses the QR ("gels") driver,
        # which ASSUMES FULL RANK. DegradedSlotEncoder pushes through a rank-10
        # projector, so X is rank-deficient by construction and lstsq returns
        # garbage there — it produced position errors of ~6.7e3 in a 1x1 world
        # before this was caught. A small ridge term makes the system
        # well-conditioned and leaves the oracle case exact (verified: residual
        # 0.00000).
        lam = 1e-6 * float(X.shape[0])
        G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
        A = torch.linalg.solve(G, X.T @ Y)
        self.register_buffer("A_read", A.float())
        resid = (X.float() @ self.A_read - flat_s)[:, :2]
        return float(resid.pow(2).sum(-1).sqrt().mean())

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(..., N, d) -> (..., N, state_dim). Analysis-only read-out."""
        if hasattr(self, "A_read"):
            zz = torch.cat([z, torch.ones_like(z[..., :1])], -1)
            return zz @ self.A_read
        return ((z - self.b) @ self.W_pinv) * self.sd + self.mu


class DegradedSlotEncoder(OracleSlotEncoder):
    """Frozen encoder with *slot bleeding* — our stand-in for SAVi.

    Before projection, each slot is mixed with the mean of the other *present*
    slots:

        x~_i = (1 - alpha) * x_i + alpha * mean_{j != i, present_j} x_j

    This deliberately violates Assumption 3 (object-aligned latents) by a
    controllable amount. `alpha = 0` recovers the oracle.

    Optionally a rank-`bottleneck` projection destroys information outright,
    modelling an encoder that simply cannot represent everything about an object.
    """

    def __init__(
        self,
        state_dim: int,
        d: int = 128,
        seed: int = 0,
        alpha: float = 0.25,
        bottleneck: int | None = None,
        scale: float = 1.0,
    ):
        super().__init__(state_dim, d, seed, scale)
        self.alpha = alpha
        if bottleneck is not None and bottleneck < state_dim:
            g = torch.Generator().manual_seed(seed + 991)
            A = torch.randn(state_dim, bottleneck, generator=g) / (state_dim ** 0.5)
            P = A @ torch.linalg.pinv(A)          # rank-`bottleneck` projector
            self.register_buffer("P", P)
        else:
            self.register_buffer("P", torch.eye(state_dim))

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._pre(state)                                     # standardised
        # `present` is channel index 5 of the packed state (see WorldConfig.state_dim)
        pres = state[..., 5:6]                                   # (...,N,1)
        n_pres = pres.sum(dim=-2, keepdim=True).clamp_min(1.0)   # (...,1,1)
        tot = (x * pres).sum(dim=-2, keepdim=True)               # (...,1,D)
        # mean over the *other* present slots
        others = (tot - x * pres) / (n_pres - pres).clamp_min(1.0)
        mixed = (1 - self.alpha) * x + self.alpha * others
        mixed = mixed * pres                                     # empty slots stay empty
        mixed = mixed @ self.P
        return mixed @ self.W + self.b


def build_encoder(kind: str, state_dim: int, d: int = 128, seed: int = 0, **kw):
    kind = kind.lower()
    if kind in ("oracle", "videosaur", "strong"):
        return OracleSlotEncoder(state_dim, d, seed)
    if kind in ("degraded", "savi", "weak"):
        return DegradedSlotEncoder(
            state_dim, d, seed,
            alpha=kw.get("alpha", 0.30),
            bottleneck=kw.get("bottleneck", 10),
        )
    raise ValueError(f"unknown encoder kind: {kind}")
