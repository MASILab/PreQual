#!/usr/bin/env python
"""
003: autoregressive FOV extension. Everything outside the acquired FOV is erased, then each missing slice is
sampled from the flow model conditioned on the n_cond slices nearer the centre (nearest last) and the target
slice's T1 (and the shell's nominal b for the multi-b model), superior end upward and inferior end downward,
each prediction conditioning the next. Both architectures are defined here (copied from models/).
Inputs : OUT/work/model_input.npz, OUT/work/meta.json, OUT/fov.json, --checkpoint PT
Writes : OUT/work/extended.npz
Invoke : python 003_extend_fov.py --out-dir OUT --checkpoint PT [--ode-steps N] [--device auto|cuda|cpu]
         [--seed 0] [--mask-pred] [--threads N]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

DEFAULT_STEPS = {"fod": 20, "tensor": 15}


# ---------------------------------------------------------------------------
# multi-b 30-direction model (models/DiFOV_flow_2d_baseGD_multiB/model.py)
# ---------------------------------------------------------------------------
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def norm(channels):
    return nn.GroupNorm(32, channels)


class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channels or channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(norm(channels), nn.SiLU(), nn.Conv2d(channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * out_channels))
        self.out_layers = nn.Sequential(norm(out_channels), nn.SiLU(), nn.Dropout(dropout),
                                        zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)))
        self.skip_connection = nn.Identity() if out_channels == channels else nn.Conv2d(channels, out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_layers[0](h) * (1 + scale) + shift
        h = self.out_layers[1:](h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.softmax(torch.einsum("bct,bcs->bts", q * scale, k * scale), dim=-1)
        return torch.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, num_head_channels=-1):
        super().__init__()
        self.num_heads = num_heads if num_head_channels == -1 else channels // num_head_channels
        self.norm = norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w)
        out = self.proj_out(self.attention(self.qkv(self.norm(x_flat))))
        return (x_flat + out).reshape(b, c, h, w)


class UNetMultiB(nn.Module):
    """Guided-diffusion UNet conditioned on (t, b) through AdaGN; forward(x, t_scaled, b_scaled)."""

    def __init__(self, in_channels, out_channels, model_channels=96, channel_mult=(1, 1, 2, 3, 4),
                 num_res_blocks=2, attention_resolutions=(8, 16), dropout=0.0, num_heads=4, num_head_channels=-1):
        super().__init__()
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        self.bvalue_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                          zero_module(nn.Linear(time_embed_dim, time_embed_dim)))
        res_block = lambda ch, out_ch: ResBlock(ch, time_embed_dim, out_ch, dropout)          # noqa: E731
        attn_block = lambda ch: AttentionBlock(ch, num_heads, num_head_channels)              # noqa: E731
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = int(mult * model_channels)
                layers = [res_block(ch, out_ch)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(attn_block(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2
        self.middle_block = TimestepEmbedSequential(res_block(ch, ch), attn_block(ch), res_block(ch, ch))
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                out_ch = int(model_channels * mult)
                layers = [res_block(ch + input_block_chans.pop(), out_ch)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(attn_block(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(norm(ch), nn.SiLU(), zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)))
        self.model_channels = model_channels

    def forward(self, x, timestep, bvalue):
        emb = (self.time_embed(timestep_embedding(timestep, self.model_channels))
               + self.bvalue_embed(timestep_embedding(bvalue, self.model_channels)))
        hs, h = [], x
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# 6-direction tensor model (models/DiFOV_flow_2d_multicond_T1/model.py)
# ---------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=time.device) * -freqs)
        emb = time[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.norm1(self.relu(self.conv1(x)))
        h = h + self.relu(self.time_mlp(t))[(...,) + (None,) * 2]
        h = self.norm2(self.relu(self.conv2(h)))
        return self.transform(h)


class UNet6Dir(nn.Module):
    """Plain 2D UNet of the 6-direction model; forward(x, t_scaled)."""

    def __init__(self, in_channels=6, out_channels=6, down_channels=(64, 128, 256, 512, 1024),
                 up_channels=(1024, 512, 256, 128, 64), time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim),
                                      nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                                    for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                                  for i in range(len(up_channels) - 1)])
        head_in = up_channels[-1] + down_channels[0]
        self.output = nn.Sequential(nn.Conv2d(head_in, up_channels[-1], 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(up_channels[-1], out_channels, 1))

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        skip0 = x
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        for up in self.ups:
            x = up(torch.cat((x, residuals.pop()), dim=1), t)
        return self.output(torch.cat((x, skip0), dim=1))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def load_model(path, kind, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    tgt_c, cond_c = ckpt["target_channels"], ckpt["cond_channels"]
    is_multib = any(k.startswith("bvalue_embed") for k in ckpt["model_state_dict"])
    if (kind == "fod") != is_multib:
        sys.exit(f"checkpoint {path} is a {'multi-b 30-dir' if is_multib else '6-dir tensor'} model but the plan "
                 f"needs the {kind} model")
    if is_multib:
        model = UNetMultiB(tgt_c + cond_c, tgt_c, model_channels=ckpt.get("model_channels", 96),
                           channel_mult=tuple(ckpt.get("channel_mult", (1, 1, 2, 3, 4))),
                           num_res_blocks=ckpt.get("num_res_blocks", 2),
                           attention_resolutions=tuple(ckpt.get("attention_resolutions", (8, 16))),
                           dropout=ckpt.get("dropout", 0.0), num_heads=ckpt.get("num_heads", 4),
                           num_head_channels=ckpt.get("num_head_channels", -1))
    else:
        model = UNet6Dir(in_channels=tgt_c + cond_c, out_channels=tgt_c)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_dir = ckpt.get("n_directions", tgt_c)
    cfg = dict(tgt_c=tgt_c, cond_c=cond_c, n_dir=n_dir, n_cond=ckpt.get("n_cond", (cond_c - 1) // n_dir),
               time_scale=float(ckpt.get("time_scale", 1000.0)), bvalue_scale=float(ckpt.get("bvalue_scale", 1.0)),
               image_size=int(ckpt.get("image_size", 128)), ckpt_steps=int(ckpt.get("n_sampling_steps", 20)))
    return model, cfg


@torch.no_grad()
def sample(model, cond, bvalue, tgt_c, n_steps, time_scale, bvalue_scale):
    """Euler integration of the rectified flow from noise (t=0) to data (t=1). -> (B, tgt_c, H, W) in [0,1]."""
    b, _, h, w = cond.shape
    x = torch.randn(b, tgt_c, h, w, device=cond.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=cond.device)
        if bvalue is None:
            v = model(torch.cat([x, cond], dim=1), t * time_scale)
        else:
            v = model(torch.cat([x, cond], dim=1), t * time_scale, bvalue * bvalue_scale)
        x = x + dt * v
    return ((x + 1) / 2).clamp(0, 1)


@torch.no_grad()
def refill(model, cfg, kind, jobs, t1, brain, n_steps, mask_pred, clip_max):
    """jobs: [dict(adc (Z,C,X,Y), b, zs ordered by distance from the known FOV, end)]; edits adc in place.
    FOD: adc in [-1,1], background -1. Tensor: adc physical x1000, background 0 (old 09 conventions)."""
    n_cond = cfg["n_cond"]
    n_max = max(len(j["zs"]) for j in jobs)
    bar = tqdm(range(n_max), desc=f"refill ({n_steps} ODE steps)", unit="slice")
    for i in bar:
        conds, targets, bs = [], [], []
        for j in jobs:
            if i >= len(j["zs"]):
                continue
            z, adc = j["zs"][i], j["adc"]
            c = adc[z - n_cond:z] if j["end"] == "superior" else adc[z + 1:z + n_cond + 1].flip(0)
            if kind == "tensor":
                c = c.clamp(0.0, clip_max) / clip_max * 2.0 - 1.0
            conds.append(torch.cat([c.reshape(-1, *adc.shape[2:]), t1[z][None]]))
            targets.append((j, z))
            bs.append(j["b"])
        cond = torch.stack(conds)
        bv = None if kind == "tensor" else torch.tensor(bs, dtype=torch.float32, device=cond.device)
        pred = sample(model, cond, bv, cfg["tgt_c"], n_steps, cfg["time_scale"], cfg["bvalue_scale"])
        for (j, z), p in zip(targets, pred):
            if kind == "fod":
                p = p * 2.0 - 1.0
                bg = -1.0
            else:
                p = p * clip_max
                bg = 0.0
            if mask_pred:
                p = torch.where(brain[z][None], p, torch.full_like(p, bg))
            j["adc"][z] = p
        bar.set_postfix(z=[z for _, z in targets])


def main(args):
    out = Path(args.out_dir)
    work = out / "work"
    meta = json.loads((work / "meta.json").read_text())
    fov = json.loads((out / "fov.json").read_text())
    kind = meta["model"]
    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: --device cuda but no CUDA device is available, running on CPU", flush=True)
            dev = "cpu"
    device = torch.device(dev)
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    model, cfg = load_model(args.checkpoint, kind, device)
    n_steps = args.ode_steps or DEFAULT_STEPS[kind]
    print(f"{kind} model on {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}) | "
          f"n_cond={cfg['n_cond']} | {n_steps} ODE steps (checkpoint default {cfg['ckpt_steps']}) | seed {args.seed}", flush=True)

    npz = np.load(work / "model_input.npz")
    t1 = torch.from_numpy(npz["t1"]).to(device)
    brain = torch.from_numpy(npz["brain"]).to(device)
    if t1.shape[1:] != (cfg["image_size"], cfg["image_size"]):
        sys.exit(f"model expects {cfg['image_size']}x{cfg['image_size']} slices, got {tuple(t1.shape[1:])}")
    z_lo, z_hi = fov["z_lo_known"], fov["z_hi_known"]
    if fov["n_known"] < cfg["n_cond"]:
        sys.exit(f"only {fov['n_known']} acquired slices, the model needs >= {cfg['n_cond']}")
    bg = -1.0 if kind == "fod" else 0.0
    jobs, arrays = [], {}
    for s in meta["shells"]:
        key = f"adc_{s['label']}"
        adc = torch.from_numpy(npz[key]).permute(0, 3, 1, 2).contiguous().to(device)    # (Z, C, X, Y)
        adc[:z_lo] = bg
        adc[z_hi + 1:] = bg
        arrays[key] = adc
        for end, zs in (("superior", fov["extend_superior"]), ("inferior", fov["extend_inferior"])):
            if zs:
                jobs.append(dict(adc=adc, b=float(s["label"]), zs=list(zs), end=end, label=s["label"]))
    t0 = time.time()
    if jobs:
        refill(model, cfg, kind, jobs, t1, brain, n_steps, args.mask_pred, float(meta["clip_max"]))
    else:
        print("nothing to extend: the T1 brain is fully inside the acquired DWI FOV", flush=True)
    np.savez(work / "extended.npz", **{k: v.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32) for k, v in arrays.items()})
    info = dict(model=kind, device=str(device), ode_steps=n_steps, n_cond=cfg["n_cond"], seed=args.seed,
                mask_pred=bool(args.mask_pred), checkpoint=str(args.checkpoint), seconds=round(time.time() - t0, 1),
                slices_per_shell={s["label"]: fov["n_extend_superior"] + fov["n_extend_inferior"] for s in meta["shells"]})
    (work / "inference.json").write_text(json.dumps(info, indent=2))
    print(f"extended {len(jobs)} job(s) in {info['seconds']} s -> {work / 'extended.npz'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ode-steps", type=int, default=None, help="default 20 (FOD) / 15 (tensor)")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-pred", action="store_true", help="background outside the T1 brain in each predicted slice")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
