"""End-to-end smoke test: physics -> encoder -> masking -> predictor -> loss.

Run:  python -m scripts.smoke
Checks the pieces agree on shapes and that the masking semantics match Sec. 4.2.
"""
import time

import torch

from cjepa.data import SlotSequenceData, build_world_data, SplitSpec
from cjepa.encoders import build_encoder
from cjepa.envs import InteractionWorld, WorldConfig
from cjepa.models import CJEPAPredictor, PredictorConfig, build_mask
from cjepa.models.losses import masked_latent_loss, slot_validity
from cjepa.models.masking import object_mask

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ---------------------------------------------------------------- 1. physics
cfg = WorldConfig(n_frames=64)
env = InteractionWorld(cfg, device=dev)
t0 = time.time()
ro = env.generate(512, seed=0)
print(f"[world] states {tuple(ro.state.shape)}  collisions {ro.events.shape[0]}  "
      f"wall {ro.wall_events.shape[0]}  ({time.time()-t0:.1f}s)")

n_ep_with_coll = len(set(ro.events[:, 0].tolist()))
print(f"[world] episodes with >=1 object-object collision: "
      f"{n_ep_with_coll}/{ro.B} ({100*n_ep_with_coll/ro.B:.1f}%)")
print(f"[world] mean collisions/episode: {ro.events.shape[0]/ro.B:.2f}")

# energy conservation sanity check (elastic collisions must conserve KE)
m = (ro.state[..., 4] ** 2)
ke = 0.5 * m * (ro.state[..., 2:4] ** 2).sum(-1)
ke = ke.sum(-1)                                   # (B, T)
drift = ((ke[:, -1] - ke[:, 0]).abs() / ke[:, 0].clamp_min(1e-9)).mean()
print(f"[world] mean |KE drift| over episode: {drift:.3%}")

# ------------------------------------------------------------- 2. masking law
M = object_mask(4, 7, 6, 10, n_mask=3, device=dev, count_mode="fixed")
assert M[:, 6:, :].all(), "all future tokens must be masked (Sec. 4.2)"
assert not M[:, 0, :].any(), "t0 identity anchor must stay visible (Eq. 3)"
per_obj = M[:, 1:6, :].any(1).sum(-1)
assert (per_obj == 3).all(), f"expected |M|=3 masked objects, got {per_obj.tolist()}"
print(f"[mask] object_mask OK — future all masked, t0 anchor visible, |M|=3 honoured")

Mu = object_mask(4096, 7, 6, 10, n_mask=4, device=dev, count_mode="uniform_upto")
cnt = Mu[:, 1:6, :].any(1).sum(-1).float()
print(f"[mask] uniform_upto |M|=4 -> mean masked objects {cnt.mean():.2f} "
      f"(expect ~2.0), max {int(cnt.max())}")

# ------------------------------------------------------------- 3. encoders
enc = build_encoder("oracle", cfg.state_dim(), 128, seed=0).to(dev)
enc.fit_normalizer(ro.state)
z = enc(ro.state[:8])
rec = enc.decode(z)
err = (rec[..., :4] - ro.state[:8, ..., :4]).abs().max()
print(f"[enc ] oracle latents {tuple(z.shape)}  decode round-trip max err {err:.2e}")

encw = build_encoder("degraded", cfg.state_dim(), 128, seed=0).to(dev)
encw.fit_normalizer(ro.state)
zw = encw(ro.state[:8])
print(f"[enc ] degraded latents {tuple(zw.shape)}  "
      f"mean |z_weak - z_oracle| = {(zw-z).abs().mean():.4f}")

# ------------------------------------------------------------- 4. predictor
pcfg = PredictorConfig(Th=6, Tp=10, n_slots=7, d_model=512, n_heads=8, head_dim=64)
model = CJEPAPredictor(pcfg).to(dev)
print(f"[pred] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
      f"d_model={pcfg.d_model} layers={pcfg.n_layers} heads={pcfg.n_heads}")

data = SlotSequenceData(ro, 6, 10, frame_skip=2, device=dev)
print(f"[data] episodes {data.B} T_sub {data.T_sub} window {data.T} "
      f"starts {data.n_starts} samples {len(data)}")

s, _, _ = data.sample_batch(32)
zz = enc(s)
mask = build_mask("object", 32, 7, 6, 10, 2, dev)
with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
    zhat, _ = model(zz, mask)
loss, lh, lf, nm = masked_latent_loss(zhat.float(), zz.float(), mask, 6,
                                      valid=slot_validity(s))
print(f"[pred] zhat {tuple(zhat.shape)} loss {loss.item():.4f} "
      f"Lh {lh.item():.4f} Lf {lf.item():.4f} masked_tokens {nm}")

# attention path used by the influence analysis
zhat2, attns = model(zz[:4], mask[:4], need_attn=True)
print(f"[pred] attention maps: {len(attns)} layers, shape {tuple(attns[0].shape)}")

# ------------------------------------------------------------- 5. inference
zr = model.rollout(zz[:, :6])
print(f"[roll] rollout {tuple(zr.shape)} (expect (32,10,7,128))")

# ------------------------------------------------------------- 6. throughput
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.time()
for _ in range(20):
    s, _, _ = data.sample_batch(256)
    zz = enc(s)
    mask = build_mask("object", 256, 7, 6, 10, 4, dev)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        zhat, _ = model(zz, mask)
        loss, *_ = masked_latent_loss(zhat.float(), zz.float(), mask, 6,
                                      valid=slot_validity(s))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
torch.cuda.synchronize() if dev == "cuda" else None
dt = (time.time() - t0) / 20
print(f"[perf] {dt*1000:.1f} ms/step @ batch 256  ->  "
      f"30 epochs x 272 steps = {dt*272*30/60:.1f} min/run")
print("SMOKE OK")
