"""
Training loop for C-JEPA / OC-JEPA.

Hyperparameters follow App. E.3 exactly where they are stated:

    "The model is trained for 30 epochs using the Adam optimizer with a batch
     size of 256. The predictor, action encoder, and proprioceptive encoder use a
     learning rate of 5e-4."

    CLEVRER: history window 6, frame skip 2, predict 10 future latents,
             mask between zero and four object slots.
    Push-T : history window 3, frame skip 5, predict 1 future latent,
             mask between zero and two object slots.

Deviation from the paper, and why: T4 GPUs are compute-capability 7.5, so we use
fp16 autocast + GradScaler rather than bf16 (bf16 is emulated and ~4x slower on
Turing). This is a numerics/throughput choice, not a method change; `--fp32`
disables it and reproduces the same curves more slowly.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import torch

from .data import SlotSequenceData
from .models.losses import masked_latent_loss, slot_validity
from .models.masking import build_mask
from .models.predictor import CJEPAPredictor, PredictorConfig


@dataclass
class TrainConfig:
    # --- masking (the independent variable of the whole study) ---------------
    n_mask: int = 0                     # |M|
    mask_strategy: str = "object"       # object | token | tube | none  (App. K)
    count_mode: str = "uniform_upto"    # App. E.3 reading vs Fig. 1 reading
    anchor: bool = True                 # keep t0 visible (Sec. 4.2)

    # --- optimisation (App. E.3) --------------------------------------------
    epochs: int = 30
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    fp16: bool = True

    seed: int = 42                      # App. H: "fixed random seed of 42"
    log_every: int = 100
    tag: str = "run"


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, encoder, data: SlotSequenceData, tcfg: TrainConfig,
             batch: int = 256, n_batches: int = 8, device="cuda"):
    """Inference-mode metrics (Sec. 4.2 'Inference'): full history, mask future only."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    lat, pos_err, n = 0.0, 0.0, 0
    Th = model.cfg.Th
    for _ in range(n_batches):
        s, _, _ = data.sample_batch(batch, generator=g)
        z = encoder(s)
        zhat = model.rollout(z[:, :Th])                       # (B, Tp, N, d)
        tgt = z[:, Th:]
        valid = slot_validity(s)[:, Th:]
        se = (zhat - tgt).pow(2).sum(-1)
        lat += (se * valid).sum().item()

        # decode to physical coordinates for an interpretable number
        ph = encoder.decode(zhat.float())
        pt = s[:, Th:]
        d = (ph[..., :2] - pt[..., :2]).pow(2).sum(-1).sqrt()
        pos_err += (d * valid).sum().item()
        n += valid.sum().item()
    model.train()
    return {"val_latent_mse": lat / max(1, n), "val_pos_err": pos_err / max(1, n)}


def train_cjepa(
    pcfg: PredictorConfig,
    tcfg: TrainConfig,
    encoder,
    train_data: SlotSequenceData,
    val_data: SlotSequenceData,
    device: str = "cuda",
    log_path: str | None = None,
    out_dir: str | None = None,
):
    torch.manual_seed(tcfg.seed)
    model = CJEPAPredictor(pcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.fp16)
    g = torch.Generator(device=device).manual_seed(tcfg.seed)

    steps_per_epoch = len(train_data) // tcfg.batch_size
    total = steps_per_epoch * tcfg.epochs
    warmup = int(tcfg.warmup_frac * total)
    Th, Tp, N = pcfg.Th, pcfg.Tp, pcfg.n_slots

    hist = []
    step = 0
    t0 = time.time()

    def log(msg):
        line = f"[{tcfg.tag}] {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(line + "\n")

    log(f"start steps/epoch={steps_per_epoch} total={total} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for ep in range(tcfg.epochs):
        run_l = run_h = run_f = 0.0
        nb = 0
        for s, _, _ in train_data.iter_epoch(tcfg.batch_size, generator=g):
            for pgrp in opt.param_groups:
                pgrp["lr"] = cosine_lr(step, total, tcfg.lr, warmup)

            with torch.no_grad():
                z = encoder(s)                       # frozen target latents
                valid = slot_validity(s)

            mask = build_mask(tcfg.mask_strategy, s.shape[0], N, Th, Tp,
                              tcfg.n_mask, device,
                              count_mode=tcfg.count_mode,
                              anchor=tcfg.anchor, generator=g)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=tcfg.fp16):
                zhat, _ = model(z, mask)
                loss, lh, lf, _ = masked_latent_loss(
                    zhat.float(), z.float(), mask, Th, valid=valid)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            run_l += loss.item(); run_h += lh.item(); run_f += lf.item(); nb += 1
            step += 1

        m = evaluate(model, encoder, val_data, tcfg, device=device)
        rec = {"epoch": ep, "loss": run_l / max(1, nb),
               "L_history": run_h / max(1, nb), "L_future": run_f / max(1, nb),
               **m, "elapsed": time.time() - t0}
        hist.append(rec)
        if ep % max(1, tcfg.epochs // 10) == 0 or ep == tcfg.epochs - 1:
            log(f"ep {ep:3d} loss {rec['loss']:.5f} "
                f"Lh {rec['L_history']:.5f} Lf {rec['L_future']:.5f} "
                f"val_mse {rec['val_latent_mse']:.5f} "
                f"pos_err {rec['val_pos_err']:.4f} ({rec['elapsed']:.0f}s)")

    result = {"config": {"predictor": asdict(pcfg), "train": asdict(tcfg)},
              "history": hist, "final": hist[-1] if hist else {}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{tcfg.tag}.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        torch.save(model.state_dict(), os.path.join(out_dir, f"{tcfg.tag}.pt"))
    return model, result
