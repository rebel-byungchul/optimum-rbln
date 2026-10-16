# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen-Image-2.1 VAE decoder rewritten as a 2D graph for images (single frame).

The original (`AutoencoderKLQwenImage21`) inherits from the Wan video VAE and uses 5D [B, C, T, H, W] tensors and feat_cache.
With T=1 the time axis is a no-op, so the same computation is done in 4D.
- CausalConv3d: already a Conv2d (time axis squeezed), so pad + conv2d as is.
- time_conv in upsample3d: on the first frame feat_cache is only set to "Rep" and the conv is not applied → omitted.
- DupUp3D(first_chunk): channel repeat + 8D view/permute + last time slot selection = constant channel selection + pixel_shuffle.
- RMS_norm: F.normalize(dim=channel) written out with rsqrt.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ops.silu (exact SiLU) gives no benefit in the VAE (PSNR 50.6→49.5 dB against the original, 9 % slower), so the default F.silu is used.
silu = F.silu


# On CR13 (dev606) a single 3×3 conv fails to compile when output channels are large and resolution is high (e.g. 96→192 @512²).
# Splitting the output channels across several convs and concatenating works (same computation). Outputs larger than this value are split.
# Enabling it for the decoder makes g3 fail instead, so it is enabled only while tracing the encoder graphs (_EncSplit).
CONV_SPLIT = {"enabled": False, "min_hw": 512 * 512, "max_out": 96}


class _EncSplit:
    def __enter__(self):
        self.prev = CONV_SPLIT["enabled"]
        CONV_SPLIT["enabled"] = True

    def __exit__(self, *a):
        CONV_SPLIT["enabled"] = self.prev


def _conv(conv, x):
    """Apply QwenImage21CausalConv3d (a Conv2d subclass) to a 4D input."""
    x = F.pad(x, conv._padding)
    w, b = conv.weight, conv.bias
    co = w.shape[0]
    if (
        CONV_SPLIT["enabled"]
        and x.shape[-2] * x.shape[-1] >= CONV_SPLIT["min_hw"]
        and co > CONV_SPLIT["max_out"]
        and w.shape[-1] > 1
    ):
        step = CONV_SPLIT["max_out"]
        return torch.cat(
            [
                F.conv2d(
                    x,
                    w[i : i + step],
                    None if b is None else b[i : i + step],
                    conv.stride,
                    0,
                    conv.dilation,
                    conv.groups,
                )
                for i in range(0, co, step)
            ],
            dim=1,
        )
    return F.conv2d(x, w, b, conv.stride, 0, conv.dilation, conv.groups)


def _rms(norm, x):
    # F.normalize(x, dim=1) = x / max(||x||_2, 1e-12), computed in fp32 like the original
    xf = x.float()
    ss = (xf * xf).sum(dim=1, keepdim=True)
    xn = (xf * torch.rsqrt(ss.clamp_min(1e-24))).to(x.dtype)
    gamma = norm.gamma.reshape(1, -1, 1, 1)
    bias = norm.bias.reshape(1, -1, 1, 1) if isinstance(norm.bias, torch.Tensor) else norm.bias
    return xn * norm.scale * gamma + bias


def _resblock(rb, x):
    h = x if isinstance(rb.conv_shortcut, nn.Identity) else _conv(rb.conv_shortcut, x)
    x = silu(_rms(rb.norm1, x))
    x = _conv(rb.conv1, x)
    x = silu(_rms(rb.norm2, x))
    x = _conv(rb.conv2, x)
    return x + h


def _attn(ab, x):
    identity = x
    B, C, H, W = x.shape
    x = _rms(ab.norm, x)
    qkv = ab.to_qkv(x).reshape(B, 1, C * 3, H * W).permute(0, 1, 3, 2)
    q, k, v = qkv.chunk(3, dim=-1)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.squeeze(1).permute(0, 2, 1).reshape(B, C, H, W)
    return ab.proj(x) + identity


def _upsample(rs, x):
    # nearest-exact with scale 2 = nearest (same indices for integer scales)
    x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    return rs.resample[1](x)


class _DupUp2D(nn.Module):
    """2D version of DupUp3D(first_chunk).

    The original uses channel repeat + view/permute, so output (c, 2h+i, 2w+j) reads input channel idx[c,i,j]. idx is constant.
    Channel gather (index_select) fails RBLN compilation at 128² and above (DEVICE_GRAPH_CONVERSION), so
    we check that idx has the form a*c + b + s*i (independent of j) and compute the same thing with strided slicing + repeat.
    """

    def __init__(self, du):
        super().__init__()
        fs, ft, oc, r = du.factor_s, du.factor_t, du.out_channels, du.repeats
        idx = torch.tensor(
            [
                [[(((c * ft + (ft - 1)) * fs + i) * fs + j) // r for j in range(fs)] for i in range(fs)]
                for c in range(oc)
            ]
        )  # [oc, fs, fs]
        if fs != 2 or not torch.equal(idx[:, :, 0], idx[:, :, 1]):
            raise NotImplementedError(f"unsupported DupUp layout: fs={fs}, idx varies across columns (j)")
        a = int(idx[1, 0, 0] - idx[0, 0, 0]) if oc > 1 else 1
        b = int(idx[0, 0, 0])
        s = int(idx[0, 1, 0] - idx[0, 0, 0])
        expect = a * torch.arange(oc)[:, None] + b + s * torch.arange(fs)[None, :]
        if not torch.equal(idx[:, :, 0], expect):
            raise NotImplementedError("DupUp channel layout is not of the form a*c+b+s*i")
        if s not in (0, 1) or (s == 1 and a != 2):
            raise NotImplementedError(f"DupUp channel layout a={a}, s={s} is not implemented yet")
        self.a, self.b, self.s, self.oc = a, b, s, oc

    def _rows(self, x, i):
        start = self.b + self.s * i
        return x[:, start : start + self.a * (self.oc - 1) + 1 : self.a]  # [B, oc, H, W]

    def forward(self, x):
        if self.s == 0:
            y = self._rows(x, 0)
            return y.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        # out[c, 2h+i, 2w+j] = x[b + a*c + s*i]; with s == 1, a == 2 this equals a 2x channel repeat + pixel_shuffle.
        # (interleaving rows via stack + reshape fails to compile at 512²)
        y = x[:, self.b : self.b + self.a * self.oc]
        return F.pixel_shuffle(y.repeat_interleave(2, dim=1), 2)


class QwenImage21VAEDecoder2D(nn.Module):
    """z [B, 64, H/16, W/16] (denormalized) → image [B, 4, H, W], clamped to [-1, 1]."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        dec = vae.decoder
        self.dup = nn.ModuleList(
            [_DupUp2D(ub.avg_shortcut) if ub.avg_shortcut is not None else nn.Identity() for ub in dec.up_blocks]
        )

    def forward(self, z):
        dec = self.vae.decoder
        x = _conv(self.vae.post_quant_conv, z)
        x = _conv(dec.conv_in, x)
        mb = dec.mid_block
        x = _resblock(mb.resnets[0], x)
        for ab, rb in zip(mb.attentions, mb.resnets[1:], strict=False):
            x = _attn(ab, x)
            x = _resblock(rb, x)
        for ub, dup in zip(dec.up_blocks, self.dup, strict=False):
            x_copy = x
            for rb in ub.resnets:
                x = _resblock(rb, x)
            if ub.upsampler is not None:
                x = _upsample(ub.upsampler, x)
            if ub.avg_shortcut is not None:
                x = x + dup(x_copy)
        x = silu(_rms(dec.norm_out, x))
        x = _conv(dec.conv_out, x)
        return x.clamp(-1.0, 1.0)


class QwenImage21VAEDecoderStage(nn.Module):
    """The decoder split into stages. stage is one of:
    "head"  : post_quant_conv + conv_in + mid_block
    "upK"   : up_blocks[K]
    "out"   : norm_out + conv_out + clamp
    """

    def __init__(self, full: "QwenImage21VAEDecoder2D", stage: str):
        super().__init__()
        self.full = full
        self.stage = stage

    def forward(self, x):
        f = self.full
        dec = f.vae.decoder
        if self.stage == "head":
            x = _conv(f.vae.post_quant_conv, x)
            x = _conv(dec.conv_in, x)
            mb = dec.mid_block
            x = _resblock(mb.resnets[0], x)
            for ab, rb in zip(mb.attentions, mb.resnets[1:], strict=False):
                x = _attn(ab, x)
                x = _resblock(rb, x)
            return x
        if self.stage.startswith("up"):
            k = int(self.stage[2:])
            ub, dup = dec.up_blocks[k], f.dup[k]
            x_copy = x
            for rb in ub.resnets:
                x = _resblock(rb, x)
            if ub.upsampler is not None:
                x = _upsample(ub.upsampler, x)
            if ub.avg_shortcut is not None:
                x = x + dup(x_copy)
            return x
        x = silu(_rms(dec.norm_out, x))
        x = _conv(dec.conv_out, x)
        return x.clamp(-1.0, 1.0)


class QwenImage21VAEDecoderPart2(nn.Module):
    """(branch, shortcut) → add → rest of the last up block → output. The last up block has no upsampler or shortcut."""

    def __init__(self, full: "QwenImage21VAEDecoder2D"):
        super().__init__()
        self.full = full
        ub = full.vae.decoder.up_blocks[-1]
        assert ub.upsampler is None and ub.avg_shortcut is None

    def forward(self, y, h):
        dec = self.full.vae.decoder
        x = y + h
        for rb in dec.up_blocks[-1].resnets[1:]:
            x = _resblock(rb, x)
        x = silu(_rms(dec.norm_out, x))
        x = _conv(dec.conv_out, x)
        return x.clamp(-1.0, 1.0)


# The CR13 compiler cannot handle masked buckets, so each resolution is compiled at its exact size. At large resolutions (2048-class),
# a single-graph decoder hits SYS_TASK_ABORTED at runtime, and compilation fails if the residual add of a channel-reducing resblock
# (576→288 @8×, 288→144 @16×) is inside the graph → split into the 4 graphs below.
#   g1: z → head ~ up_blocks[2]                       → x (input of up_blocks[3])
#   g2: x → (branch, shortcut) of up_blocks[3].resnets[0]
#   g3: (branch, shortcut, x) → rest of up_blocks[3] → (branch, shortcut) of up_blocks[4].resnets[0]
#   g4: (branch, shortcut) → rest of up_blocks[4] → norm_out → conv_out → clamp


class VAEChainG1(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full

    def forward(self, z):
        f = self.full
        x = QwenImage21VAEDecoderStage(f, "head")(z)
        for k in range(3):
            x = QwenImage21VAEDecoderStage(f, f"up{k}")(x)
        return x


class VAEChainG2(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full

    def forward(self, x):
        rb = self.full.vae.decoder.up_blocks[3].resnets[0]
        h = x if isinstance(rb.conv_shortcut, nn.Identity) else _conv(rb.conv_shortcut, x)
        y = _conv(rb.conv1, silu(_rms(rb.norm1, x)))
        y = _conv(rb.conv2, silu(_rms(rb.norm2, y)))
        return y, h


class VAEChainG3(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full

    def forward(self, y, h, x_copy):
        f, dec = self.full, self.full.vae.decoder
        ub, dup = dec.up_blocks[3], f.dup[3]
        x = y + h
        for rb in ub.resnets[1:]:
            x = _resblock(rb, x)
        x = _upsample(ub.upsampler, x)
        x = x + dup(x_copy)
        rb = dec.up_blocks[4].resnets[0]
        h2 = x if isinstance(rb.conv_shortcut, nn.Identity) else _conv(rb.conv_shortcut, x)
        y2 = _conv(rb.conv1, silu(_rms(rb.norm1, x)))
        y2 = _conv(rb.conv2, silu(_rms(rb.norm2, y2)))
        return y2, h2


class VAEChainG4(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full

    def forward(self, y, h):
        return QwenImage21VAEDecoderPart2(self.full)(y, h)


class _DecStep(nn.Module):
    def __init__(self, full, fn):
        super().__init__()
        self.full, self.fn = full, fn

    def forward(self, *args):
        return self.fn(self.full, *args)


def _f3a(f, y, h):  # add of up_blocks[3].resnets[0] → resnets[1]
    return _resblock(f.vae.decoder.up_blocks[3].resnets[1], y + h)


def _f3b(f, x):  # up_blocks[3].resnets[2:]
    for rb in f.vae.decoder.up_blocks[3].resnets[2:]:
        x = _resblock(rb, x)
    return x


def _f3c(f, x, x_copy):  # upsampler + duplicated-channel shortcut of up_blocks[3]
    return _upsample(f.vae.decoder.up_blocks[3].upsampler, x) + f.dup[3](x_copy)


def _f3d(f, x):  # (branch, shortcut) of up_blocks[4].resnets[0] (channel-reducing: the add stays outside)
    rb = f.vae.decoder.up_blocks[4].resnets[0]
    h = x if isinstance(rb.conv_shortcut, nn.Identity) else _conv(rb.conv_shortcut, x)
    y = _conv(rb.conv1, silu(_rms(rb.norm1, x)))
    y = _conv(rb.conv2, silu(_rms(rb.norm2, y)))
    return y, h


def _f4a(f, y, h):
    return _resblock(f.vae.decoder.up_blocks[4].resnets[1], y + h)


def _f4b(f, x):
    for rb in f.vae.decoder.up_blocks[4].resnets[2:]:
        x = _resblock(rb, x)
    return x


def _f4c(f, x):
    dec = f.vae.decoder
    return _conv(dec.conv_out, silu(_rms(dec.norm_out, x))).clamp(-1.0, 1.0)


# Fine decoder chain: (graph name, module factory, input names, output names). Same ops as VAEChainG1..G4, cut into more graphs.
# Init/Intermediate scratch adds up over the layers of a graph and a CR13 device holds at most ~64 GiB of it (shared with the
# DiT), so at the 2048 class the high-resolution part (g3: 28.7 GiB, g4: 21.6 GiB at 2528x1696) is split per resblock.
class _Stage(nn.Module):
    def __init__(self, full, stage):
        super().__init__()
        self.full, self.stage = full, stage

    def forward(self, x):
        return QwenImage21VAEDecoderStage(self.full, self.stage)(x)


DECODER_PROGRAM = [
    # f1 (= VAEChainG1) split per stage: as one graph it needed 9.5-10 GiB of chiplet 0 at the 2048 class
    ("f1h", lambda f: _Stage(f, "head"), ["z"], ["x0"]),
    ("f1u0", lambda f: _Stage(f, "up0"), ["x0"], ["x1u"]),
    ("f1u1", lambda f: _Stage(f, "up1"), ["x1u"], ["x2u"]),
    ("f1u2", lambda f: _Stage(f, "up2"), ["x2u"], ["x"]),
    ("f2", VAEChainG2, ["x"], ["y", "h"]),
    ("f3a", lambda f: _DecStep(f, _f3a), ["y", "h"], ["x1"]),
    ("f3b", lambda f: _DecStep(f, _f3b), ["x1"], ["x2"]),
    ("f3c", lambda f: _DecStep(f, _f3c), ["x2", "x"], ["x3"]),
    ("f3d", lambda f: _DecStep(f, _f3d), ["x3"], ["y2", "h2"]),
    ("f4a", lambda f: _DecStep(f, _f4a), ["y2", "h2"], ["x4"]),
    ("f4b", lambda f: _DecStep(f, _f4b), ["x4"], ["x5"]),
    ("f4c", lambda f: _DecStep(f, _f4c), ["x5"], ["out"]),
]


# Everything after the mid block (f1u1 .. f4c) has only local ops (convs, per-pixel norms, nearest upsample, pixel shuffle),
# so it can run in row tiles: each tile gets TILE_HALO extra latent rows on both sides and keeps only its own rows, which
# gives exactly the untiled result. The receptive radius of that tail is about 6.7 latent rows (3.5 from up_blocks[1] at 2x,
# 1.75 from up_blocks[2] at 4x, ~1.4 from the 8x/16x stages); the halo is 8. Tiles are planned in latent rows and every
# tensor is sliced at its own scale.
TILED_STEPS = ("f1u1", "f1u2", "f2", "f3a", "f3b", "f3c", "f3d", "f4a", "f4b", "f4c")
TILE_HALO = 8


def tile_plan(rows, n, halo=TILE_HALO):
    """Row windows for n tiles over `rows` rows: (T, [(start, keep0, keep1)]) with one common window height T."""
    if n <= 1:
        return rows, [(0, 0, rows)]
    T = min(rows, -(-rows // n) + 2 * halo)
    plan = []
    for i in range(n):
        b0, b1 = rows * i // n, rows * (i + 1) // n
        s = min(max(b0 - halo, 0), rows - T)
        plan.append((s, b0 - s, b1 - s))
    return T, plan


# Coarse chain for sizes that need no tiling (up to the 1024 class): fewer graph boundaries, so fewer host round trips.
DECODER_PROGRAM_COARSE = [
    ("g1", VAEChainG1, ["z"], ["x"]),
    ("g2", VAEChainG2, ["x"], ["y", "h"]),
    ("g3", VAEChainG3, ["y", "h", "x"], ["y2", "h2"]),
    ("g4", VAEChainG4, ["y2", "h2"], ["out"]),
]


def run_program(program, graphs, env, n_tiles=1, use=None, tiled=TILED_STEPS, halo=TILE_HALO, base="z", unit=1):
    """Run a chain program; `use(name)` is an optional context manager around each graph (load on demand).

    Steps in `tiled` run in row tiles planned in units of `unit` rows of env[base] (latent rows for the decoder, 16 pixel
    rows for the encoder), with `halo` units on both sides; every tensor is sliced at its own scale, and the outputs of
    the last tiled step are stitched back (each tile keeps only its own rows)."""
    from contextlib import nullcontext

    use = use or (lambda name: nullcontext())
    if n_tiles > 1:
        H = env[base].shape[2] // unit
        T, plan = tile_plan(H, n_tiles, halo)
    tiles, last_tiled = None, None
    for name, _, ins, outs in program:
        is_tiled = name in tiled and n_tiles > 1
        if tiles is not None and not is_tiled:
            _stitch(env, tiles, last_tiled, T)
            tiles = None
        with use(name):
            if is_tiled:
                if tiles is None:  # split the inputs of the first tiled step into row windows (each at its own scale)
                    tiles = []
                    for st, k0, k1 in plan:
                        tenv = {}
                        for k in ins:
                            sc = env[k].shape[2] // H
                            tenv[k] = env[k][:, :, st * sc : (st + T) * sc].contiguous()
                        tiles.append((tenv, k0, k1))
                for tenv, _, _ in tiles:
                    res = graphs[name](*[tenv[k] for k in ins])
                    tenv.update(zip(outs, res if isinstance(res, list | tuple) else (res,), strict=True))
                last_tiled = outs
            else:
                res = graphs[name](*[env[k] for k in ins])
                env.update(zip(outs, res if isinstance(res, list | tuple) else (res,), strict=True))
    if tiles is not None:
        _stitch(env, tiles, last_tiled, T)
    return env


def _stitch(env, tiles, keys, T):
    for key in keys:
        sc = tiles[0][0][key].shape[2] // T
        env[key] = torch.cat([t[key][:, :, sc * k0 : sc * k1] for t, k0, k1 in tiles], dim=2)


# ------------------------------------------------------------------------------------------ encoder (condition image)
# Original encode (first chunk, T=1) in 2D. AvgDown3D(first chunk) prepends a zero frame (pad on T) and averages channel groups,
# which equals a constant 1×1 linear map after pixel_unshuffle (the zero frame contributes 0). time_conv in downsample3d is unused
# on the first chunk (feat_cache is None → only stored). The output is the 64-channel mean after quant_conv (argmax = mode).


class _AvgDown2D(nn.Module):
    def __init__(self, ad):
        super().__init__()
        ft, fs, C, oc, g = ad.factor_t, ad.factor_s, ad.in_channels, ad.out_channels, ad.group_size
        self.fs = fs
        k = fs * fs
        w = torch.zeros(oc, C * k)
        for c in range(C):
            for q in range(k):
                # original flattened channel index: c·(ft·k) + t·k + q; the real frame is t = ft-1 (earlier t are zero padding)
                flat = c * ft * k + (ft - 1) * k + q
                w[flat // g, c * k + q] += 1.0 / g
        # pixel_unshuffle is unsupported (aten::pixel_unshuffle) → 1×1 conv on each stride-2 slice, then sum
        ws = [w[:, q::k][:, :, None, None].contiguous() for q in range(k)]  # q = i·fs + j
        for q, wq in enumerate(ws):
            self.register_buffer(f"w{q}", wq, persistent=False)
        self.k = k

    def forward(self, x):
        if self.fs == 1:
            return F.conv2d(x, self.w0.to(x.dtype))
        fs = self.fs
        out = None
        for i in range(fs):
            for j in range(fs):
                q = i * fs + j
                y = F.conv2d(x[:, :, i::fs, j::fs], getattr(self, f"w{q}").to(x.dtype))
                out = y if out is None else out + y
        return out


class QwenImage21VAEEncoder2D(nn.Module):
    """Image [1, 4, H, W] ([-1,1], RGBA) → latent mean [1, 64, H/16, W/16] (before normalization)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        enc = vae.encoder
        self.avg = nn.ModuleList([_AvgDown2D(db.avg_shortcut) for db in enc.down_blocks])

    def forward(self, *args):
        with _EncSplit():
            return self._forward(*args)

    def _forward(self, x):
        enc = self.vae.encoder
        x = _conv(enc.conv_in, x)
        for db, avg in zip(enc.down_blocks, self.avg, strict=False):
            x_copy = x
            for rb in db.resnets:
                x = _resblock(rb, x)
            if db.downsampler is not None:
                x = db.downsampler.resample(x)  # ZeroPad2d((0,1,0,1)) + Conv2d stride 2
            x = x + avg(x_copy)
        mb = enc.mid_block
        x = _resblock(mb.resnets[0], x)
        for ab, rb in zip(mb.attentions, mb.resnets[1:], strict=False):
            x = _attn(ab, x)
            x = _resblock(rb, x)
        x = _conv(enc.conv_out, silu(_rms(enc.norm_out, x)))
        x = _conv(self.vae.quant_conv, x)
        return x[:, : self.vae.config.z_dim]


class QwenImage21VAEEncoderStage(nn.Module):
    """The encoder split into stages: "in" (conv_in) | "downK" (down_blocks[K]) | "mid" | "out" (norm_out+conv_out+quant_conv+mean)"""

    def __init__(self, enc2d: "QwenImage21VAEEncoder2D", stage: str):
        super().__init__()
        self.e, self.stage = enc2d, stage

    def forward(self, x):
        e = self.e
        enc = e.vae.encoder
        if self.stage == "in":
            return _conv(enc.conv_in, x)
        if self.stage.startswith("down"):
            k = int(self.stage[4:])
            db, avg = enc.down_blocks[k], e.avg[k]
            x_copy = x
            for rb in db.resnets:
                x = _resblock(rb, x)
            if db.downsampler is not None:
                x = db.downsampler.resample(x)
            return x + avg(x_copy)
        if self.stage == "mid":
            mb = enc.mid_block
            x = _resblock(mb.resnets[0], x)
            for ab, rb in zip(mb.attentions, mb.resnets[1:], strict=False):
                x = _attn(ab, x)
                x = _resblock(rb, x)
            return x
        x = _conv(enc.conv_out, silu(_rms(enc.norm_out, x)))
        x = _conv(e.vae.quant_conv, x)
        return x[:, : e.vae.config.z_dim]


# encoder chain: the graph is cut at the residual add of the channel-changing resblock (down_blocks[1].resnets[0], 96→192 @ H/2).
# Encoder chain as a program. e1..e3 (full and half resolution) have only local ops, so they can run in row tiles planned
# in units of 16 pixel rows with a 2-unit halo (their receptive radius is about 20 pixels); e4 (low resolution, mid-block
# attention) runs whole. Tiling is used when a size does not compile in one piece (shape-specific compiler errors).
ENC_TILED_STEPS = ("e1", "e2", "e3")
ENC_TILE_HALO = 2
ENC_TILE_UNIT = 16


class VAEEncChainE1(nn.Module):
    def __init__(self, enc2d):
        super().__init__()
        self.e = enc2d

    def forward(self, *args):
        with _EncSplit():
            return self._forward(*args)

    def _forward(self, x):
        x = QwenImage21VAEEncoderStage(self.e, "in")(x)
        return QwenImage21VAEEncoderStage(self.e, "down0")(x)


class VAEEncChainE2(nn.Module):
    def __init__(self, enc2d):
        super().__init__()
        self.e = enc2d

    def forward(self, *args):
        with _EncSplit():
            return self._forward(*args)

    def _forward(self, x):
        rb = self.e.vae.encoder.down_blocks[1].resnets[0]
        h = x if isinstance(rb.conv_shortcut, nn.Identity) else _conv(rb.conv_shortcut, x)
        y = _conv(rb.conv1, silu(_rms(rb.norm1, x)))
        y = _conv(rb.conv2, silu(_rms(rb.norm2, y)))
        return y, h


class VAEEncChainE3(nn.Module):
    """Add + rest of down_blocks[1] (+ downsample, shortcut mean)"""

    def __init__(self, enc2d):
        super().__init__()
        self.e = enc2d

    def forward(self, *args):
        with _EncSplit():
            return self._forward(*args)

    def _forward(self, y, h, x_copy):
        e = self.e
        db, avg = e.vae.encoder.down_blocks[1], e.avg[1]
        x = y + h
        for rb in db.resnets[1:]:
            x = _resblock(rb, x)
        if db.downsampler is not None:
            x = db.downsampler.resample(x)
        return x + avg(x_copy)


class VAEEncChainE4(nn.Module):
    """down_blocks[2:] + mid + out"""

    def __init__(self, enc2d):
        super().__init__()
        self.e = enc2d

    def forward(self, *args):
        with _EncSplit():
            return self._forward(*args)

    def _forward(self, x):
        e = self.e
        for k in range(2, len(e.vae.encoder.down_blocks)):
            x = QwenImage21VAEEncoderStage(e, f"down{k}")(x)
        x = QwenImage21VAEEncoderStage(e, "mid")(x)
        return QwenImage21VAEEncoderStage(e, "out")(x)


ENCODER_PROGRAM = [
    ("e1", VAEEncChainE1, ["x"], ["x1"]),
    ("e2", VAEEncChainE2, ["x1"], ["y", "h"]),
    ("e3", VAEEncChainE3, ["y", "h", "x1"], ["x3"]),
    ("e4", VAEEncChainE4, ["x3"], ["z"]),
]
