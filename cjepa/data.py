"""
Dataset construction: simulate once, cache states, encode on the fly.

App. E.3 trains "on pre-extracted object embeddings", so the encoder is applied
outside the training loop in the paper. We cache the cheaper *states* instead and
apply the (frozen, deterministic, matmul-only) encoder per batch. That is
numerically identical to pre-extraction and lets us swap the strong/weak encoder
without regenerating physics — which matters, because the encoder-quality x |M|
interaction (Tab. 1) is one of the results we are trying to reproduce.

Temporal subsampling follows App. D.1/E.3: "frames are temporally subsampled with
stride two" for CLEVRER, "a frame skip of five" for Push-T.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .envs.interaction_world import InteractionWorld, WorldConfig, Rollout


@dataclass
class SplitSpec:
    n_episodes: int
    seed: int


class SlotSequenceData:
    """Holds simulated episodes and serves (Th + Tp)-length latent windows."""

    def __init__(
        self,
        rollout: Rollout,
        Th: int,
        Tp: int,
        frame_skip: int = 2,
        device: str = "cuda",
    ):
        self.rollout = rollout
        self.Th, self.Tp, self.skip = Th, Tp, frame_skip
        self.device = device
        # (B, T_sub, N, D)
        self.states = rollout.state[:, ::frame_skip].contiguous().to(device)
        self.present = rollout.present.to(device)
        self.B, self.T_sub, self.N, self.D = self.states.shape
        self.T = Th + Tp
        self.n_starts = max(1, self.T_sub - self.T + 1)

    def __len__(self) -> int:
        return self.B * self.n_starts

    def window_states(self, ep_idx: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Gather (b, start:start+T) windows -> (batch, T, N, D)."""
        offs = torch.arange(self.T, device=self.device)
        tt = start[:, None] + offs[None, :]                     # (b, T)
        return self.states[ep_idx[:, None], tt]                 # (b, T, N, D)

    def sample_batch(self, batch: int, generator=None):
        ep = torch.randint(0, self.B, (batch,), device=self.device, generator=generator)
        st = torch.randint(0, self.n_starts, (batch,), device=self.device, generator=generator)
        return self.window_states(ep, st), ep, st

    def iter_epoch(self, batch: int, generator=None, shuffle: bool = True):
        """Deterministic full pass over every (episode, window-start) pair."""
        total = len(self)
        idx = (torch.randperm(total, device=self.device, generator=generator)
               if shuffle else torch.arange(total, device=self.device))
        for i in range(0, total - batch + 1, batch):
            sel = idx[i : i + batch]
            ep, st = sel // self.n_starts, sel % self.n_starts
            yield self.window_states(ep, st), ep, st


def build_world_data(
    cfg: WorldConfig,
    splits: dict[str, SplitSpec],
    Th: int,
    Tp: int,
    frame_skip: int = 2,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> dict[str, SlotSequenceData]:
    env = InteractionWorld(cfg, device=device)
    out = {}
    for name, spec in splits.items():
        path = (os.path.join(cache_dir, f"world_{name}_{spec.n_episodes}_{spec.seed}.pt")
                if cache_dir else None)
        if path and os.path.exists(path):
            blob = torch.load(path, map_location=device, weights_only=False)
            ro = Rollout(blob["state"], blob["present"], blob["events"],
                         blob["wall_events"], cfg)
        else:
            ro = env.generate(spec.n_episodes, seed=spec.seed)
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"state": ro.state.cpu(), "present": ro.present.cpu(),
                            "events": ro.events.cpu(),
                            "wall_events": ro.wall_events.cpu()}, path)
            ro = ro.to(device)
        out[name] = SlotSequenceData(ro, Th, Tp, frame_skip, device)
    return out
