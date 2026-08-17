"""
InteractionWorld — a CLEVRER-analogue multi-object collision environment.

WHY THIS ENVIRONMENT
--------------------
C-JEPA's central claim (Sec. 4.2, Thm. 1) is that masking an object's *entire
history* — keeping only an identity anchor at t0 — makes "interaction reasoning
functionally necessary" for minimising the loss, because the masked object's
state can no longer be recovered by trivial temporal interpolation of its own
past.

For that claim to be testable, the dynamics must have exactly two regimes:

  1. **Self-dynamics** (ballistic motion between collisions), which a predictor
     can nail with a linear-extrapolation *shortcut* and no interaction
     reasoning at all.
  2. **Interaction-dependent dynamics** (elastic collisions), where an object's
     future depends on *which other object it hit and when*.

Bouncing balls in a box give precisely this. Crucially — and unlike CLEVRER —
we get for free:

  * a **ground-truth temporal interaction graph** (who collided with whom, when),
    which lets us directly validate the paper's `influence neighborhood`
    (Def. 1 / Cor. 1). The authors explicitly list this as *not done*:
    "while we formally characterize influence neighborhoods, we do not directly
     validate them on datasets with explicit temporal causal graphs, leaving
     this to future work" (Sec. 7, Limitations).
  * exact **counterfactual rollouts** (re-simulate with object k deleted), which
    give us CLEVRER-style counterfactual questions with *exact* answers rather
    than annotator labels.

CLEVRER PARITY
--------------
CLEVRER has at most 6 simultaneously visible objects and the paper therefore
uses N = 7 slots, "where one slot implicitly captures background or empty
regions" (App. C.1). We mirror that exactly: up to 6 real objects padded out to
7 slots with explicit `present=0` empty slots.

All physics is deterministic given the initial state, which is what makes the
counterfactual branch well-defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class WorldConfig:
    n_slots: int = 7            # App. C.1: 7 slots for CLEVRER (6 objects + bg)
    max_objects: int = 6        # CLEVRER: max 6 simultaneously visible objects
    min_objects: int = 4
    n_colors: int = 8           # CLEVRER has 8 colours
    n_shapes: int = 3           # cube / sphere / cylinder

    box: float = 1.0            # world is [0, box]^2
    r_min: float = 0.055
    r_max: float = 0.085
    speed_min: float = 0.012
    speed_max: float = 0.028

    n_frames: int = 48          # frames stored per episode
    substeps: int = 6           # physics substeps per stored frame (anti-tunnelling)
    restitution: float = 1.0    # 1.0 = perfectly elastic

    seed: int = 0

    def state_dim(self) -> int:
        # [x, y, vx, vy, r, present] + colour one-hot + shape one-hot
        return 6 + self.n_colors + self.n_shapes


# --------------------------------------------------------------------------- #
# Rollout container
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    """A batch of simulated episodes.

    state   : (B, T, N, state_dim)  per-frame object states
    present : (B, N)                which slots hold a real object
    events  : (E, 4) long tensor of collision events [b, t, i, j] with i < j.
              Object-object collisions only; wall bounces are recorded in
              `wall_events` because they are *self*-dynamics, not interactions.
    """

    state: torch.Tensor
    present: torch.Tensor
    events: torch.Tensor
    wall_events: torch.Tensor
    cfg: WorldConfig = field(default_factory=WorldConfig)

    @property
    def B(self) -> int:
        return self.state.shape[0]

    @property
    def T(self) -> int:
        return self.state.shape[1]

    @property
    def N(self) -> int:
        return self.state.shape[2]

    def to(self, device) -> "Rollout":
        return Rollout(
            self.state.to(device),
            self.present.to(device),
            self.events.to(device),
            self.wall_events.to(device),
            self.cfg,
        )

    def interaction_graph(self, t_lo: int, t_hi: int) -> torch.Tensor:
        """Ground-truth undirected interaction graph over the window [t_lo, t_hi).

        Returns (B, N, N) float tensor, 1.0 where objects i and j collided at
        least once inside the window. This is the empirical stand-in for the
        paper's influence neighborhood N_t(i) (Def. 1) — see cjepa/eval/influence.py.
        """
        B, N = self.B, self.N
        g = torch.zeros(B, N, N, device=self.state.device)
        if self.events.numel() == 0:
            return g
        ev = self.events
        m = (ev[:, 1] >= t_lo) & (ev[:, 1] < t_hi)
        ev = ev[m]
        if ev.numel() == 0:
            return g
        g[ev[:, 0], ev[:, 2], ev[:, 3]] = 1.0
        g[ev[:, 0], ev[:, 3], ev[:, 2]] = 1.0
        return g


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class InteractionWorld:
    """Deterministic, GPU-vectorised elastic-collision simulator.

    Everything is batched over episodes: one `simulate` call advances *all*
    episodes together, so generating a 4k-episode dataset is a couple of seconds
    on a T4.
    """

    def __init__(self, cfg: WorldConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

    # ---------------------------------------------------------------- sampling
    def sample_initial(self, B: int, generator: torch.Generator | None = None):
        """Rejection-sample non-overlapping initial configurations."""
        cfg, dev = self.cfg, self.device
        N = cfg.n_slots

        def rnd(*shape):
            return torch.rand(*shape, device=dev, generator=generator)

        # how many real objects in each episode
        n_obj = torch.randint(
            cfg.min_objects, cfg.max_objects + 1, (B,), device=dev, generator=generator
        )
        present = (torch.arange(N, device=dev)[None, :] < n_obj[:, None]).float()

        r = cfg.r_min + (cfg.r_max - cfg.r_min) * rnd(B, N)
        r = r * present  # empty slots have zero radius

        # --- rejection sampling for non-overlapping placement -----------------
        pos = torch.zeros(B, N, 2, device=dev)
        for i in range(N):
            for _attempt in range(200):
                cand = r[:, i : i + 1] + rnd(B, 2) * (cfg.box - 2 * r[:, i : i + 1])
                if i == 0:
                    pos[:, i] = cand
                    break
                d = torch.linalg.norm(pos[:, :i] - cand[:, None, :], dim=-1)  # (B,i)
                need = (r[:, :i] + r[:, i : i + 1]) * 1.06
                bad = ((d < need) & (present[:, :i] > 0)).any(dim=-1)  # (B,)
                # accept candidate wherever it is valid, keep retrying elsewhere
                fill = (~bad) & (pos[:, i].abs().sum(-1) == 0)
                pos[fill, i] = cand[fill]
                if not (pos[:, i].abs().sum(-1) == 0).any():
                    break
            # any episode that never found a slot: drop that object
            never = pos[:, i].abs().sum(-1) == 0
            if never.any():
                present[never, i] = 0.0
                r[never, i] = 0.0
                pos[never, i] = 0.5

        theta = rnd(B, N) * 2 * math.pi
        speed = cfg.speed_min + (cfg.speed_max - cfg.speed_min) * rnd(B, N)
        vel = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * speed[..., None]
        vel = vel * present[..., None]

        color = torch.randint(0, cfg.n_colors, (B, N), device=dev, generator=generator)
        shape = torch.randint(0, cfg.n_shapes, (B, N), device=dev, generator=generator)
        return pos, vel, r, present, color, shape

    # ------------------------------------------------------------------ physics
    def _pack(self, pos, vel, r, present, color, shape) -> torch.Tensor:
        cfg = self.cfg
        oh_c = torch.nn.functional.one_hot(color, cfg.n_colors).float() * present[..., None]
        oh_s = torch.nn.functional.one_hot(shape, cfg.n_shapes).float() * present[..., None]
        return torch.cat(
            [pos, vel, r[..., None], present[..., None], oh_c, oh_s], dim=-1
        )

    def simulate(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        r: torch.Tensor,
        present: torch.Tensor,
        color: torch.Tensor,
        shape: torch.Tensor,
        n_frames: int | None = None,
    ) -> Rollout:
        """Run deterministic physics and record the interaction graph."""
        cfg, dev = self.cfg, self.device
        T = n_frames or cfg.n_frames
        B, N = pos.shape[0], pos.shape[1]
        dt = 1.0 / cfg.substeps

        pos, vel = pos.clone(), vel.clone()
        mass = (r ** 2).clamp_min(1e-8)  # 2-D "area" mass; empty slots ~0

        iu, ju = torch.triu_indices(N, N, offset=1, device=dev)  # (P,)
        P = iu.numel()

        frames, events, wall_events = [], [], []

        for t in range(T):
            frames.append(self._pack(pos, vel, r, present, color, shape))

            for _ in range(cfg.substeps):
                pos = pos + vel * dt

                # ---- wall collisions (self-dynamics: recorded but not "interaction")
                for ax in (0, 1):
                    lo = pos[..., ax] < r
                    hi = pos[..., ax] > cfg.box - r
                    hit = (lo | hi) & (present > 0)
                    if hit.any():
                        b_idx, n_idx = hit.nonzero(as_tuple=True)
                        wall_events.append(
                            torch.stack(
                                [b_idx, torch.full_like(b_idx, t), n_idx,
                                 torch.full_like(b_idx, ax)], dim=-1)
                        )
                    pos[..., ax] = torch.where(lo, 2 * r - pos[..., ax], pos[..., ax])
                    pos[..., ax] = torch.where(
                        hi, 2 * (cfg.box - r) - pos[..., ax], pos[..., ax]
                    )
                    vel[..., ax] = torch.where(lo | hi, -vel[..., ax], vel[..., ax])

                # ---- pairwise elastic collisions -----------------------------
                # Resolved pair-by-pair (P is tiny: 21 for N=7) but batched over B.
                for p in range(P):
                    i, j = int(iu[p]), int(ju[p])
                    both = (present[:, i] > 0) & (present[:, j] > 0)
                    if not both.any():
                        continue
                    d = pos[:, i] - pos[:, j]                       # (B,2)
                    dist = torch.linalg.norm(d, dim=-1).clamp_min(1e-8)
                    rad = r[:, i] + r[:, j]
                    dv = vel[:, i] - vel[:, j]
                    approaching = (d * dv).sum(-1) < 0
                    hit = both & (dist < rad) & approaching
                    if not hit.any():
                        continue

                    b_idx = hit.nonzero(as_tuple=True)[0]
                    events.append(
                        torch.stack(
                            [b_idx,
                             torch.full_like(b_idx, t),
                             torch.full_like(b_idx, i),
                             torch.full_like(b_idx, j)], dim=-1)
                    )

                    n_hat = d / dist[:, None]
                    mi, mj = mass[:, i], mass[:, j]
                    v_rel = (dv * n_hat).sum(-1)
                    imp = (1 + cfg.restitution) * v_rel / (mi + mj)
                    dvi = -(imp * mj)[:, None] * n_hat
                    dvj = (imp * mi)[:, None] * n_hat
                    h = hit[:, None].float()
                    vel[:, i] = vel[:, i] + h * dvi
                    vel[:, j] = vel[:, j] + h * dvj

                    # positional de-overlap so pairs don't stick together
                    overlap = (rad - dist).clamp_min(0.0)
                    corr = 0.5 * overlap[:, None] * n_hat * h
                    pos[:, i] = pos[:, i] + corr
                    pos[:, j] = pos[:, j] - corr

        state = torch.stack(frames, dim=1)  # (B,T,N,D)
        ev = torch.cat(events, 0) if events else torch.zeros(0, 4, dtype=torch.long, device=dev)
        we = (torch.cat(wall_events, 0) if wall_events
              else torch.zeros(0, 4, dtype=torch.long, device=dev))
        return Rollout(state, present, ev.long(), we.long(), cfg)

    # ------------------------------------------------------------------- public
    def generate(self, B: int, seed: int | None = None) -> Rollout:
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.seed if seed is None else seed)
        init = self.sample_initial(B, generator=g)
        return self.simulate(*init)

    def counterfactual(
        self,
        rollout_init: tuple,
        delete: torch.Tensor,
        n_frames: int | None = None,
    ) -> Rollout:
        """Re-simulate from the *same* initial condition with object `delete[b]`
        removed from episode b.

        This is the do-operator that defines a CLEVRER counterfactual question:
        "if object k were removed, would A and B still collide?" Because the
        physics is deterministic, the answer is exact.
        """
        pos, vel, r, present, color, shape = [x.clone() for x in rollout_init]
        B = pos.shape[0]
        b_idx = torch.arange(B, device=pos.device)
        valid = delete >= 0
        present[b_idx[valid], delete[valid]] = 0.0
        r[b_idx[valid], delete[valid]] = 0.0
        vel[b_idx[valid], delete[valid]] = 0.0
        return self.simulate(pos, vel, r, present, color, shape, n_frames=n_frames)
