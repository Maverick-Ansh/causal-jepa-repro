"""
The C-JEPA predictor  f : Zbar_Tbar -> Zhat_Tbar   (Eq. 4).

ARCHITECTURE (App. E.1, E.2)
----------------------------
"The predictor f is a ViT-style masked transformer with bidirectional attention,
 enabling joint inference over masked object tokens across the history window and
 future horizon."  (Sec. 4.2)

"We maintain object latent states with slot dimensionality 128 throughout the
 whole experiment. Future prediction is performed by a Transformer-based
 predictor with six layers, 16 attention heads, head dimension 64, and an MLP
 hidden dimension of 2048."  (App. E.2)

App. E.1 explains *why* it is bidirectional rather than autoregressive:

    "Autoregressive predictors impose a sequential dependency structure that
     conditions each prediction on previously generated tokens, which can bias
     learning toward local self-dynamics. In contrast, masked prediction allows
     the model to attend jointly to the entire history window and infer masked
     object states in parallel."

TOKENISATION
------------
Entity tokens are  Z_t = {S_t, U_t}  (Sec. 4.2): the N object slots plus any
auxiliary variables. We flatten history+future into a sequence of length T*N and
append A auxiliary tokens per timestep, giving L = T*(N + A).

THE ONE THING THAT IS EASY TO GET WRONG
---------------------------------------
There is **no positional encoding along the entity (slot) dimension** — the paper
is explicit about this, because slot representations are permutation-equivariant
and adding slot indices would break that symmetry:

    "...the permutation equivariance of slot-based representations with respect to
     object ordering, which motivates omitting positional encodings along the
     entity dimension, consistent with prior work."

The consequence is load-bearing: a masked token carries object identity *only*
through phi(z^i_t0). Ablating the anchor (`anchor=False`) should therefore be
catastrophic, not merely worse — we test that.

Temporal position, by contrast, *is* encoded (e_tau).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictorConfig:
    slot_dim: int = 128          # App. E.2
    d_model: int = 512           # paper: 16 heads x 64 = 1024; see note in README
    n_heads: int = 8             # paper: 16
    head_dim: int = 64           # App. E.2
    n_layers: int = 6            # App. E.2
    mlp_hidden: int = 2048       # App. E.2
    dropout: float = 0.0
    Th: int = 6                  # App. E.3: history window 6 for CLEVRER
    Tp: int = 10                 # App. E.3: predict 10 future latents for CLEVRER
    n_slots: int = 7             # App. C.1
    n_aux: int = 0               # actions / proprioception tokens per timestep
    aux_dim: int = 0

    @property
    def T(self) -> int:
        return self.Th + self.Tp

    def __post_init__(self):
        # keep head_dim honoured: d_model must equal n_heads * head_dim
        if self.d_model != self.n_heads * self.head_dim:
            self.d_model = self.n_heads * self.head_dim


class Attention(nn.Module):
    """Bidirectional MHA with an optional slow path that returns attention maps.

    The slow path exists for cjepa/eval/influence.py, which reads cross-slot
    attention as the empirical proxy for the influence neighborhood N_t(i)
    (Cor. 1). SPARTAN (Lei et al., 2025) uses the same proxy and the paper
    follows it in App. J.
    """

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x, need_attn: bool = False):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B, h, L, dh)
        if not need_attn:
            o = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop if self.training else 0.0
            )
            attn = None
        else:
            a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            attn = a.softmax(dim=-1)                  # (B, h, L, L)
            o = attn @ v
        o = o.transpose(1, 2).reshape(B, L, self.h * self.dh)
        return self.proj(o), attn


class Block(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.d_model),
        )

    def forward(self, x, need_attn: bool = False):
        h, attn = self.attn(self.n1(x), need_attn)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, attn


class CJEPAPredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # visible slot tokens
        self.in_proj = nn.Linear(cfg.slot_dim, D)
        # Eq. 3: phi, the linear projection applied to the identity anchor
        self.phi = nn.Linear(cfg.slot_dim, D)
        # Eq. 3: the "learnable embedding" component of e_tau
        self.mask_embed = nn.Parameter(torch.zeros(D))
        # Eq. 3: the "temporal positional encoding" component of e_tau.
        # NOTE: indexed by time only. There is deliberately no slot embedding.
        self.time_embed = nn.Parameter(torch.zeros(cfg.T, D))

        if cfg.n_aux > 0:
            # App. D.3: auxiliaries get their own lightweight temporal embedder
            self.aux_proj = nn.Conv1d(cfg.aux_dim, D, kernel_size=3, padding=1)
            self.aux_type = nn.Parameter(torch.zeros(cfg.n_aux, D))

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, cfg.slot_dim)

        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_embed, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        z: torch.Tensor,             # (B, T, N, slot_dim)  encoder targets
        mask: torch.Tensor,          # (B, T, N)  bool, True == masked
        aux: torch.Tensor | None = None,   # (B, T, n_aux, aux_dim)
        need_attn: bool = False,
    ):
        """Build Zbar (Eq. 3), run f (Eq. 4), return Zhat over slot tokens."""
        cfg = self.cfg
        B, T, N, _ = z.shape

        # ---- Eq. 3: masked tokens are phi(anchor) + e_tau -------------------- #
        anchor = z[:, 0]                                  # (B, N, slot_dim)  z^i_t0
        vis_tok = self.in_proj(z)                         # (B, T, N, D)
        msk_tok = self.phi(anchor)[:, None].expand(B, T, N, cfg.d_model) \
                  + self.mask_embed
        x = torch.where(mask[..., None], msk_tok, vis_tok)

        # temporal positional encoding (shared by masked and visible tokens)
        x = x + self.time_embed[None, :T, None, :]
        x = x.reshape(B, T * N, cfg.d_model)

        # ---- auxiliary entity tokens (never masked; Fig. 1) ------------------ #
        if cfg.n_aux > 0 and aux is not None:
            a = aux.permute(0, 2, 3, 1).reshape(B * cfg.n_aux, cfg.aux_dim, T)
            a = self.aux_proj(a).reshape(B, cfg.n_aux, cfg.d_model, T)
            a = a.permute(0, 3, 1, 2)                     # (B, T, n_aux, D)
            a = a + self.aux_type[None, None] + self.time_embed[None, :T, None, :]
            x = torch.cat([x, a.reshape(B, T * cfg.n_aux, cfg.d_model)], dim=1)

        attns = []
        for blk in self.blocks:
            x, at = blk(x, need_attn)
            if need_attn:
                attns.append(at)

        x = self.norm(x)
        slot_out = x[:, : T * N].reshape(B, T, N, cfg.d_model)
        zhat = self.out_proj(slot_out)                     # (B, T, N, slot_dim)
        return (zhat, attns) if need_attn else (zhat, None)

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def rollout(self, z_hist: torch.Tensor, aux: torch.Tensor | None = None):
        """Inference mode (Sec. 4.2, 'Inference').

            "At inference time, C-JEPA performs forward latent prediction
             following Eq. 4, with a fully observable history and masking applied
             only to future tokens."

        z_hist: (B, Th, N, slot_dim).  Returns (B, Tp, N, slot_dim).
        """
        cfg = self.cfg
        B, Th, N, d = z_hist.shape
        assert Th == cfg.Th, f"expected history {cfg.Th}, got {Th}"
        z = torch.zeros(B, cfg.T, N, d, device=z_hist.device, dtype=z_hist.dtype)
        z[:, :Th] = z_hist
        mask = torch.zeros(B, cfg.T, N, dtype=torch.bool, device=z_hist.device)
        mask[:, Th:] = True                                # future only
        zhat, _ = self.forward(z, mask, aux)
        return zhat[:, Th:]
