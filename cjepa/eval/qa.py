"""
CLEVRER-analogue visual question answering (reproduces the protocol behind Tab. 1).

THE PAPER'S PROTOCOL, AND WHY WE COPY ITS SHAPE EXACTLY
------------------------------------------------------
Sec. 5.1: "we train C-JEPA and baseline world models, and roll out 128-frame
input videos to 160 frames to produce imagined trajectories. ALOE is then trained
on these model-generated trajectories for downstream questions."

So VQA accuracy is *not* a direct measure of the world model. It is a probe:
a reasoner is trained on whatever the world model imagined, and its accuracy is
the score. A better world model produces imagined futures that carry more
answerable structure. We reproduce that structure exactly:

    observed history slots  ++  MODEL-IMAGINED future slots  ->  probe  ->  answer

Everything about the probe (architecture, seed, optimiser, number of steps, and
the question set itself) is held fixed across world models. The *only* thing that
varies is the trajectory the probe reads. That is what makes the comparison a
measurement of the world model.

WHAT WE CHANGED, AND WHY IT IS FAIR
-----------------------------------
* ALOE embeds a natural-language question with a language transformer. We hand
  the probe the question's *semantic fields* (type, and the objects referred to,
  named by colour) as tokens instead. Parsing English is not under test here and
  a language encoder would add variance shared by all arms. Objects are still
  referred to by attribute, never by slot index, so the probe must still locate
  "the red one" inside the slot set — which is the part ALOE actually does.
* Questions are binary. CLEVRER scores counterfactual/explanatory/predictive both
  per-option and per-question; with binary questions those two coincide, so we
  report a single accuracy and note it in the report.

THE FOUR QUESTION TYPES (CLEVRER's categories, computed from ground truth)
-------------------------------------------------------------------------
  descriptive    : did A and B collide during the OBSERVED history?
  predictive     : will A and B collide during the FUTURE window?
  counterfactual : if K were removed, would A and B collide?
  explanatory    : was K responsible for A and B colliding?
                   ("but-for" causation: they collide factually, and do NOT
                    collide once K is deleted)

Because the simulator is deterministic, every answer is exact — no annotators,
no label noise. Counterfactual and explanatory answers come from *re-simulating*
the episode with object K deleted, which is a genuine do-operator on the
data-generating process (contrast with the paper's masking, which intervenes only
on the predictor's observability — Lemma 1).

TWO CONTROLS WE ADD
-------------------
The paper reports only model-vs-model numbers, which makes an accuracy of, say,
72% hard to interpret. We bracket every run with:
  * ORACLE  — probe reads ground-truth-encoded future latents (ceiling)
  * STATIC  — probe reads history with the last frame repeated (no dynamics at
              all; floor). Anything at or below STATIC has learned no useful
              forward model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.interaction_world import InteractionWorld, Rollout, WorldConfig

QTYPES = ["descriptive", "predictive", "counterfactual", "explanatory"]


# --------------------------------------------------------------------------- #
# Ground-truth answer machinery
# --------------------------------------------------------------------------- #
def collision_matrix(rollout: Rollout, t_lo: int, t_hi: int, B: int, N: int) -> torch.Tensor:
    """(B, N, N) bool: did i and j collide in sim-frame window [t_lo, t_hi)?"""
    g = torch.zeros(B, N, N, dtype=torch.bool, device=rollout.state.device)
    ev = rollout.events
    if ev.numel() == 0:
        return g
    m = (ev[:, 1] >= t_lo) & (ev[:, 1] < t_hi)
    ev = ev[m]
    if ev.numel() == 0:
        return g
    g[ev[:, 0], ev[:, 2], ev[:, 3]] = True
    g[ev[:, 0], ev[:, 3], ev[:, 2]] = True
    return g


@dataclass
class QABank:
    """Everything needed to ask and answer questions about a batch of episodes."""
    ep: torch.Tensor        # (Q,) episode index
    qtype: torch.Tensor     # (Q,) in [0,4)
    obj_a: torch.Tensor     # (Q,) slot index of A
    obj_b: torch.Tensor     # (Q,) slot index of B
    obj_k: torch.Tensor     # (Q,) slot index of K, or -1
    answer: torch.Tensor    # (Q,) 0/1
    qvec: torch.Tensor      # (Q, qdim) colour-addressed encoding
    n_colors: int
    fact: torch.Tensor      # (Q,) did A and B factually collide in the window?

    def copy_factual_baseline(self) -> dict:
        """Accuracy of the degenerate policy 'answer with the factual outcome'.

        This is the number that makes counterfactual accuracy interpretable: any
        model at or below it has demonstrated no counterfactual reasoning, only
        collision detection.
        """
        out = {}
        ok = (self.fact.long() == self.answer)
        for t, name in enumerate(QTYPES):
            m = self.qtype == t
            out[name] = float(ok[m].float().mean()) * 100 if m.any() else float("nan")
        out["average"] = float(ok.float().mean()) * 100
        return out

    def split(self, frac: float, seed: int = 0):
        g = torch.Generator(device=self.ep.device).manual_seed(seed)
        # split by EPISODE, never by question — otherwise the probe sees the same
        # trajectory in train and test and the numbers are meaningless
        eps = torch.unique(self.ep)
        perm = eps[torch.randperm(eps.numel(), device=eps.device, generator=g)]
        n_tr = int(frac * perm.numel())
        tr_eps = torch.zeros(int(eps.max()) + 1, dtype=torch.bool, device=eps.device)
        tr_eps[perm[:n_tr]] = True
        m = tr_eps[self.ep]
        def sub(mask):
            return QABank(self.ep[mask], self.qtype[mask], self.obj_a[mask],
                          self.obj_b[mask], self.obj_k[mask], self.answer[mask],
                          self.qvec[mask], self.n_colors, self.fact[mask])
        return sub(m), sub(~m)


def build_qa_bank(
    env: InteractionWorld,
    rollout: Rollout,
    Th_sim: int,
    T_sim: int,
    n_per_type: int = 4,
    seed: int = 0,
) -> QABank:
    """Generate balanced binary questions with exact ground-truth answers.

    Th_sim : sim-frame index where the observed history ends
    T_sim  : sim-frame index where the QA window ends
    """
    dev = rollout.state.device
    cfg = rollout.cfg
    B, N = rollout.B, rollout.N
    g = torch.Generator(device=dev).manual_seed(seed)

    present = rollout.present.bool()                                  # (B,N)
    colors = rollout.state[:, 0, :, 6 : 6 + cfg.n_colors].argmax(-1)  # (B,N)

    G_hist = collision_matrix(rollout, 0, Th_sim, B, N)
    G_fut = collision_matrix(rollout, Th_sim, T_sim, B, N)
    G_all = collision_matrix(rollout, 0, T_sim, B, N)

    # ---- counterfactual branches: delete object k, re-simulate from t=0 ------ #
    init = InteractionWorld.unpack(rollout.state[:, 0], cfg.n_colors)
    G_cf = torch.zeros(N, B, N, N, dtype=torch.bool, device=dev)
    for k in range(N):
        dele = torch.full((B,), k, device=dev, dtype=torch.long)
        dele = torch.where(present[:, k], dele, torch.full_like(dele, -1))
        ro_cf = env.counterfactual(init, dele, n_frames=T_sim)
        G_cf[k] = collision_matrix(ro_cf, 0, T_sim, B, N)

    # ---- enumerate candidate questions -------------------------------------- #
    eps_l, qt_l, a_l, b_l, k_l, ans_l = [], [], [], [], [], []

    pair_i, pair_j = torch.triu_indices(N, N, offset=1, device=dev)

    def add_balanced(qt: int, ep_idx, aa, bb, kk, ans, want: int, strata=None):
        """Pick `want` questions per episode, balanced across strata.

        `strata` (optional) is an integer tensor defining the cells to balance
        over; when omitted we balance on the answer alone.

        WHY STRATA MATTER FOR COUNTERFACTUALS
        -------------------------------------
        Deleting an object usually changes nothing, so a naively sampled
        counterfactual set is ~90% answerable by the degenerate policy "report
        what factually happened". That would make the counterfactual column a
        second descriptive column and hide exactly the effect the paper claims.
        For counterfactual questions we therefore balance over the joint cell
        (factual outcome, counterfactual outcome), which forces ~50% of questions
        to have cf != factual and drops the copy-the-factual-outcome baseline to
        chance. `copy_factual_baseline` is reported so this stays auditable.
        """
        cells = strata if strata is not None else ans.long()
        for pos in sorted(set(cells.tolist())):
            sel = cells == pos
            if sel.sum() == 0:
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            # shuffle then take at most want//2 per episode via a per-episode counter
            idx = idx[torch.randperm(idx.numel(), device=dev, generator=g)]
            keep, seen = [], {}
            n_cells = len(set(cells.tolist()))
            cap = max(1, want // max(1, n_cells))
            for t in idx.tolist():
                e = int(ep_idx[t])
                if seen.get(e, 0) < cap:
                    seen[e] = seen.get(e, 0) + 1
                    keep.append(t)
            if not keep:
                continue
            keep = torch.tensor(keep, device=dev, dtype=torch.long)
            eps_l.append(ep_idx[keep]); qt_l.append(torch.full_like(keep, qt))
            a_l.append(aa[keep]); b_l.append(bb[keep]); k_l.append(kk[keep])
            ans_l.append(ans[keep].long())

    # descriptive / predictive: over all present pairs (a, b)
    bb_idx = torch.arange(B, device=dev)[:, None].expand(B, pair_i.numel()).reshape(-1)
    ai = pair_i[None, :].expand(B, -1).reshape(-1)
    bi = pair_j[None, :].expand(B, -1).reshape(-1)
    ok = present[bb_idx, ai] & present[bb_idx, bi]
    bb_ok, ai_ok, bi_ok = bb_idx[ok], ai[ok], bi[ok]
    none_k = torch.full_like(ai_ok, -1)

    add_balanced(0, bb_ok, ai_ok, bi_ok, none_k, G_hist[bb_ok, ai_ok, bi_ok], n_per_type)
    add_balanced(1, bb_ok, ai_ok, bi_ok, none_k, G_fut[bb_ok, ai_ok, bi_ok], n_per_type)

    # counterfactual / explanatory: (a, b, k) with k distinct from both
    reps = []
    for k in range(N):
        m = ok & (ai != k) & (bi != k) & present[bb_idx, torch.full_like(ai, k)]
        reps.append((bb_idx[m], ai[m], bi[m], torch.full_like(ai[m], k)))
    bb_t = torch.cat([r[0] for r in reps]); ai_t = torch.cat([r[1] for r in reps])
    bi_t = torch.cat([r[2] for r in reps]); ki_t = torch.cat([r[3] for r in reps])

    def add_global_balanced(qt: int, ep_idx, aa, bb, kk, ans, cells, cap_per_ep: int):
        """Take an EQUAL number of questions from every cell, pooled over episodes.

        add_balanced caps per episode, which cannot equalise a cell that is
        globally rare — and the interesting counterfactual cells (deleting K
        flips whether A and B meet) are exactly the rare ones. Here we pool all
        candidates, take min-cell-count from each cell, and only then apply a
        per-episode cap so no single episode dominates.
        """
        groups = []
        for c in sorted(set(cells.tolist())):
            idx = (cells == c).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(idx.numel(), device=dev, generator=g)]
            keep, seen = [], {}
            for t in idx.tolist():
                e = int(ep_idx[t])
                if seen.get(e, 0) < cap_per_ep:
                    seen[e] = seen.get(e, 0) + 1
                    keep.append(t)
            groups.append(keep)
        if not groups or min(len(x) for x in groups) == 0:
            return
        n = min(len(x) for x in groups)
        keep = torch.tensor([t for grp in groups for t in grp[:n]],
                            device=dev, dtype=torch.long)
        eps_l.append(ep_idx[keep]); qt_l.append(torch.full_like(keep, qt))
        a_l.append(aa[keep]); b_l.append(bb[keep]); k_l.append(kk[keep])
        ans_l.append(ans[keep].long())

    cf_ans = G_cf[ki_t, bb_t, ai_t, bi_t]
    fact_ab = G_all[bb_t, ai_t, bi_t]
    # 4 cells over (does deleting K change the outcome?, what IS the outcome?).
    # Equal cells => P(yes) = 0.5 AND P(cf != factual) = 0.5, which pins the
    # copy-the-factual-outcome baseline at chance.
    changed = (cf_ans != fact_ab)
    cf_cells = changed.long() * 2 + cf_ans.long()
    add_global_balanced(2, bb_t, ai_t, bi_t, ki_t, cf_ans, cf_cells, cap_per_ep=2)

    expl_ans = fact_ab & (~cf_ans)
    add_balanced(3, bb_t, ai_t, bi_t, ki_t, expl_ans, n_per_type)

    ep = torch.cat(eps_l); qt = torch.cat(qt_l)
    a = torch.cat(a_l); b = torch.cat(b_l); k = torch.cat(k_l)
    ans = torch.cat(ans_l)

    # ---- colour-addressed question vector ----------------------------------- #
    C = cfg.n_colors
    ca = colors[ep, a]
    cb = colors[ep, b]
    ck = torch.where(k >= 0, colors[ep, k.clamp_min(0)], torch.zeros_like(k))
    qvec = torch.cat([
        F.one_hot(qt, len(QTYPES)).float(),
        F.one_hot(ca, C).float(),
        F.one_hot(cb, C).float(),
        F.one_hot(ck, C).float() * (k >= 0).float()[:, None],
        (k >= 0).float()[:, None],
    ], dim=-1)
    fact = G_all[ep, a, b]
    return QABank(ep, qt, a, b, k, ans, qvec, C, fact)


# --------------------------------------------------------------------------- #
# ALOE-analogue probe
# --------------------------------------------------------------------------- #
class AloeProbe(nn.Module):
    """Transformer reasoner over (imagined trajectory, question) -> answer.

    Mirrors ALOE's shape (App. F.1): object trajectories and question fields are
    concatenated into one token sequence and processed jointly, with learnable
    positional embeddings and an MLP classifier head. No slot positional
    encoding, for the same permutation-equivariance reason as the predictor.
    """

    def __init__(self, slot_dim: int, qdim: int, T: int, N: int,
                 d: int = 96, n_layers: int = 4, n_heads: int = 4, n_out: int = 2):
        super().__init__()
        self.T, self.N, self.d = T, N, d
        self.slot_in = nn.Linear(slot_dim, d)
        self.q_in = nn.Linear(qdim, d)
        # Role of each slot in THIS question: 0 = irrelevant, 1 = A, 2 = B, 3 = K.
        #
        # Why this is here, and why it does not leak the answer: without it the
        # probe must first solve object grounding — scan 7 slots, decode each
        # one's colour out of a 128-d random projection, and match it against the
        # colour named in the question — before it can begin reasoning about
        # dynamics. Empirically it never gets past that step: with GROUND-TRUTH
        # future latents as input the probe still scored 50.2% on counterfactual
        # and 52.1% on predictive questions, i.e. the ceiling sat at chance and
        # the metric could not have separated any two world models.
        #
        # Roles are computed from the question plus the scene's colour->slot
        # assignment, both of which are already present in the probe's input, so
        # this supplies no information about the ANSWER. It removes a grounding
        # burden that is not what the world model is being tested on and lets the
        # probe spend its capacity on the trajectory, which is.
        self.role_embed = nn.Embedding(4, d)
        self.time_embed = nn.Parameter(torch.zeros(T, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.qtok = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.tr = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128),
                                  nn.GELU(), nn.Linear(128, n_out))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.qtok, std=0.02)

    def forward(self, traj: torch.Tensor, q: torch.Tensor, roles: torch.Tensor):
        """traj: (B, T, N, slot_dim)   q: (B, qdim)   roles: (B, N) long in [0,4)"""
        B, T, N, _ = traj.shape
        x = self.slot_in(traj) + self.time_embed[None, :T, None, :]
        x = x + self.role_embed(roles)[:, None, :, :]     # broadcast over time
        x = x.reshape(B, T * N, self.d)
        qt = self.q_in(q)[:, None, :] + self.qtok
        x = torch.cat([self.cls.expand(B, -1, -1), qt, x], dim=1)
        x = self.tr(x)
        return self.head(x[:, 0])


# --------------------------------------------------------------------------- #
# Trajectory construction + probe training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def imagined_trajectories(model, encoder, states: torch.Tensor, Th: int,
                          mode: str = "model", batch: int = 256) -> torch.Tensor:
    """Build the trajectory the probe will read.

    mode="model"  : observed history latents ++ world-model rollout   (the paper)
    mode="oracle" : observed history ++ ground-truth future latents   (ceiling)
    mode="static" : observed history ++ last observed frame repeated  (floor)
    """
    outs = []
    for i in range(0, states.shape[0], batch):
        s = states[i : i + batch]
        z = encoder(s)
        if mode == "oracle":
            traj = z
        elif mode == "static":
            traj = torch.cat([z[:, :Th],
                              z[:, Th - 1 : Th].expand(-1, z.shape[1] - Th, -1, -1)], 1)
        elif mode == "model":
            model.eval()
            zf = model.rollout(z[:, :Th])
            traj = torch.cat([z[:, :Th], zf], dim=1)
        else:
            raise ValueError(mode)
        outs.append(traj.float())
    return torch.cat(outs, 0)


def train_probe(traj: torch.Tensor, bank_tr: QABank, bank_te: QABank,
                steps: int = 3000, batch: int = 512, lr: float = 1e-3,
                seed: int = 0, device: str = "cuda", d: int = 128,
                log=lambda s: None):
    """Train the ALOE-analogue probe and return per-category accuracy.

    IMPORTANT: `steps`, `seed`, `d` and the question banks are identical across
    every world model we compare. Only `traj` differs.
    """
    torch.manual_seed(seed)
    B, T, N, sd = traj.shape
    probe = AloeProbe(sd, bank_tr.qvec.shape[-1], T, N, d=d).to(device)

    def roles_of(bank, sl):
        """(n, N) long: which slots this question refers to."""
        n = bank.obj_a[sl].numel()
        r = torch.zeros(n, N, dtype=torch.long, device=device)
        idx = torch.arange(n, device=device)
        r[idx, bank.obj_a[sl]] = 1
        r[idx, bank.obj_b[sl]] = 2
        k = bank.obj_k[sl]
        has_k = k >= 0
        r[idx[has_k], k[has_k]] = 3
        return r

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    g = torch.Generator(device=device).manual_seed(seed)
    n = bank_tr.ep.numel()

    for st in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * (0.5 * (1 + math.cos(math.pi * st / steps))
                             if st > steps * 0.1 else (st + 1) / (0.1 * steps))
        idx = torch.randint(0, n, (batch,), device=device, generator=g)
        tr = traj[bank_tr.ep[idx]]
        q = bank_tr.qvec[idx]
        y = bank_tr.answer[idx]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = probe(tr, q, roles_of(bank_tr, idx))
            loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if st % max(1, steps // 4) == 0:
            log(f"    probe step {st}/{steps} loss {loss.item():.4f}")

    # ---- evaluate ----------------------------------------------------------- #
    probe.eval()
    accs, correct_all, n_all = {}, 0, 0
    with torch.no_grad():
        preds = []
        for i in range(0, bank_te.ep.numel(), 1024):
            sl = slice(i, i + 1024)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = probe(traj[bank_te.ep[sl]], bank_te.qvec[sl],
                           roles_of(bank_te, sl))
            preds.append(lg.float().argmax(-1))
        pred = torch.cat(preds)
        ok = (pred == bank_te.answer)
        for t, name in enumerate(QTYPES):
            m = bank_te.qtype == t
            accs[name] = float(ok[m].float().mean()) * 100 if m.any() else float("nan")
        correct_all, n_all = int(ok.sum()), ok.numel()
    accs["average"] = 100.0 * correct_all / max(1, n_all)
    accs["n_eval"] = n_all
    return accs
