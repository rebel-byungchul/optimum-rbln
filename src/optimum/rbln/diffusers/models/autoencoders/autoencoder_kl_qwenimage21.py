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

import copy
import gc
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import rebel
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage21
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import nn

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLQwenImage21Config


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

# ------------------------------------------------------------------------------------------------ 2D building blocks
# The original VAE inherits the Wan video VAE: 5D tensors and a feature cache over frames. An image is a single frame,
# for which the time axis is a no-op, so the same computation runs on 4D tensors here.


# On RBLN-CR13 a 3x3 conv with many output channels at a high resolution fails to compile in the encoder (e.g. 96 → 192
# channels at 512x512, DEVICE_GRAPH_CONVERSION); the same conv split into groups of output channels compiles. The
# encoder's convs carry `split_out_channels` (see `_split_encoder_convs`); the decoder graphs compile without it (and
# with it, one of them does not).
_SPLIT_MIN_PIXELS, _SPLIT_OUT_CHANNELS = 512 * 512, 96


def _split_encoder_convs(vae: AutoencoderKLQwenImage21) -> AutoencoderKLQwenImage21:
    for module in vae.encoder.modules():
        if isinstance(module, nn.Conv2d):
            module.split_out_channels = _SPLIT_OUT_CHANNELS
    return vae


def _conv(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A `QwenImage21CausalConv3d` (a Conv2d with its own padding) on a 4D input."""
    x = F.pad(x, conv._padding)
    weight, bias, out_channels = conv.weight, conv.bias, conv.weight.shape[0]
    step = getattr(conv, "split_out_channels", None)
    if not (step and x.shape[-2] * x.shape[-1] >= _SPLIT_MIN_PIXELS and out_channels > step and weight.shape[-1] > 1):
        return F.conv2d(x, weight, bias, conv.stride, 0, conv.dilation, conv.groups)
    return torch.cat(
        [
            F.conv2d(
                x,
                weight[i : i + step],
                None if bias is None else bias[i : i + step],
                conv.stride,
                0,
                conv.dilation,
                conv.groups,
            )
            for i in range(0, out_channels, step)
        ],
        dim=1,
    )


def _rms(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # F.normalize over the channels, computed in float32 like the original.
    xf = x.float()
    normalized = (xf * torch.rsqrt((xf * xf).sum(dim=1, keepdim=True).clamp_min(1e-24))).to(x.dtype)
    bias = norm.bias.reshape(1, -1, 1, 1) if isinstance(norm.bias, torch.Tensor) else norm.bias
    return normalized * norm.scale * norm.gamma.reshape(1, -1, 1, 1) + bias


def _resblock_branch(block: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shortcut = x if isinstance(block.conv_shortcut, nn.Identity) else _conv(block.conv_shortcut, x)
    x = _conv(block.conv1, F.silu(_rms(block.norm1, x)))
    return _conv(block.conv2, F.silu(_rms(block.norm2, x))), shortcut


def _resblock(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    branch, shortcut = _resblock_branch(block, x)
    return branch + shortcut


def _attention(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    qkv = block.to_qkv(_rms(block.norm, x)).reshape(batch, 1, channels * 3, height * width).permute(0, 1, 3, 2)
    q, k, v = qkv.chunk(3, dim=-1)
    out = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(batch, channels, height, width)
    return block.proj(out) + x


def _upsample(resample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # nearest-exact at scale 2 is nearest
    return resample.resample[1](x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


class _DupUp2D(nn.Module):
    """`QwenImage21DupUp3D` on the first frame: output channel `c` at `(2h+i, 2w+j)` reads a constant input channel.
    A channel gather fails to compile at large sizes, so the layouts of the checkpoint (`a*c + b + s*i`) are computed
    with strided slices, a repeat and a pixel shuffle."""

    def __init__(self, dup: nn.Module):
        super().__init__()
        fs, ft, oc, r = dup.factor_s, dup.factor_t, dup.out_channels, dup.repeats
        index = torch.tensor(
            [
                [[(((c * ft + (ft - 1)) * fs + i) * fs + j) // r for j in range(fs)] for i in range(fs)]
                for c in range(oc)
            ]
        )
        if fs != 2 or not torch.equal(index[:, :, 0], index[:, :, 1]):
            raise NotImplementedError(f"Unsupported DupUp layout (factor_s={fs}).")
        a = int(index[1, 0, 0] - index[0, 0, 0]) if oc > 1 else 1
        b, s = int(index[0, 0, 0]), int(index[0, 1, 0] - index[0, 0, 0])
        if not torch.equal(index[:, :, 0], a * torch.arange(oc)[:, None] + b + s * torch.arange(fs)[None, :]) or not (
            s == 0 or (s == 1 and a == 2)
        ):
            raise NotImplementedError("Unsupported DupUp channel layout.")
        self.a, self.b, self.s, self.oc = a, b, s, oc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.s == 0:
            y = x[:, self.b : self.b + self.a * (self.oc - 1) + 1 : self.a]
            return y.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        # a == 2, s == 1: a 2x channel repeat and a pixel shuffle
        return F.pixel_shuffle(x[:, self.b : self.b + self.a * self.oc].repeat_interleave(2, dim=1), 2)


class _AvgDown2D(nn.Module):
    """`QwenImage21AvgDown3D` on the first frame (a zero frame is prepended, so it is a constant linear map after a
    pixel unshuffle), as 1x1 convs over the stride-`factor_s` slices (pixel_unshuffle does not compile)."""

    def __init__(self, avg: nn.Module):
        super().__init__()
        ft, fs, channels, oc, group = avg.factor_t, avg.factor_s, avg.in_channels, avg.out_channels, avg.group_size
        k = fs * fs
        weight = torch.zeros(oc, channels * k)
        for c in range(channels):
            for q in range(k):
                weight[(c * ft * k + (ft - 1) * k + q) // group, c * k + q] += 1.0 / group
        for q in range(k):
            self.register_buffer(f"w{q}", weight[:, q::k][:, :, None, None].contiguous(), persistent=False)
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fs = self.fs
        return sum(
            F.conv2d(x[:, :, i::fs, j::fs], getattr(self, f"w{i * fs + j}").to(x.dtype))
            for i in range(fs)
            for j in range(fs)
        )


class _VAE2D(nn.Module):
    """The VAE modules and the 2D replacements of its first-frame shortcut layers, shared by the chain steps."""

    def __init__(self, vae: AutoencoderKLQwenImage21):
        super().__init__()
        self.vae = vae
        self.dup = nn.ModuleList(
            [
                _DupUp2D(ub.avg_shortcut) if ub.avg_shortcut is not None else nn.Identity()
                for ub in vae.decoder.up_blocks
            ]
        )
        self.avg = nn.ModuleList([_AvgDown2D(db.avg_shortcut) for db in vae.encoder.down_blocks])

    def mid(self, mid_block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = _resblock(mid_block.resnets[0], x)
        for attention, block in zip(mid_block.attentions, mid_block.resnets[1:], strict=False):
            x = _resblock(block, _attention(attention, x))
        return x

    def up(self, k: int, x: torch.Tensor) -> torch.Tensor:
        return self.up_upsample(k, self.up_resnets(k, x), x)

    def up_resnets(self, k: int, x: torch.Tensor) -> torch.Tensor:
        for block in self.vae.decoder.up_blocks[k].resnets:
            x = _resblock(block, x)
        return x

    def up_upsample(self, k: int, x: torch.Tensor, x_copy: torch.Tensor) -> torch.Tensor:
        up_block = self.vae.decoder.up_blocks[k]
        if up_block.upsampler is not None:
            x = _upsample(up_block.upsampler, x)
        return x + self.dup[k](x_copy) if up_block.avg_shortcut is not None else x

    def down(self, k: int, x: torch.Tensor) -> torch.Tensor:
        down_block, x_copy = self.vae.encoder.down_blocks[k], x
        for block in down_block.resnets:
            x = _resblock(block, x)
        if down_block.downsampler is not None:
            x = down_block.downsampler.resample(x)  # ZeroPad2d((0, 1, 0, 1)) + stride-2 conv
        return x + self.avg[k](x_copy)

    def decoder_out(self, x: torch.Tensor) -> torch.Tensor:
        decoder = self.vae.decoder
        return _conv(decoder.conv_out, F.silu(_rms(decoder.norm_out, x))).clamp(-1.0, 1.0)


class _Step(nn.Module):
    def __init__(self, vae2d: _VAE2D, fn: Callable):
        super().__init__()
        self.vae2d, self.fn = vae2d, fn

    def forward(self, *args):
        return self.fn(self.vae2d, *args)


def _decoder_head(f, z):
    decoder = f.vae.decoder
    return f.mid(decoder.mid_block, _conv(decoder.conv_in, _conv(f.vae.post_quant_conv, z)))


def _decoder_up012(f, x):
    for k in range(3):
        x = f.up(k, x)
    return x


def _decoder_up3_branch(f, x):
    # The residual add of a channel-changing resblock is kept outside the graph (inside, the graph fails to compile).
    return _resblock_branch(f.vae.decoder.up_blocks[3].resnets[0], x)


def _decoder_up3_rest(f, y, h, x_copy):
    up_block = f.vae.decoder.up_blocks[3]
    x = y + h
    for block in up_block.resnets[1:]:
        x = _resblock(block, x)
    x = _upsample(up_block.upsampler, x) + f.dup[3](x_copy)
    return _resblock_branch(f.vae.decoder.up_blocks[4].resnets[0], x)


def _decoder_up4_rest(f, y, h):
    x = y + h
    for block in f.vae.decoder.up_blocks[4].resnets[1:]:
        x = _resblock(block, x)
    return f.decoder_out(x)


def _up3_resnet(i):
    return lambda f, *xs: _resblock(f.vae.decoder.up_blocks[3].resnets[i], sum(xs))


def _up4_resnet(i):
    return lambda f, *xs: _resblock(f.vae.decoder.up_blocks[4].resnets[i], sum(xs))


def _decoder_up3_rest_resnets(f, x):
    for block in f.vae.decoder.up_blocks[3].resnets[2:]:
        x = _resblock(block, x)
    return x


def _decoder_up4_rest_resnets(f, x):
    for block in f.vae.decoder.up_blocks[4].resnets[2:]:
        x = _resblock(block, x)
    return x


# A chain program: (step name, function, input names, output names). Untiled sizes (up to the 1024 class) run the
# coarse decoder; tiled ones the fine one, whose graphs are smaller (the device scratch adds up over the layers of a
# graph). Every step after the mid block has only local ops (convs, per-pixel norms, nearest upsampling, pixel
# shuffles), so those run in row tiles with a halo of TILE_HALO latent rows, which gives exactly the untiled result.
DECODER_PROGRAM = [
    ("head_up012", lambda f, z: _decoder_up012(f, _decoder_head(f, z)), ["z"], ["x"]),
    ("up3_branch", _decoder_up3_branch, ["x"], ["y", "h"]),
    ("up3_rest", _decoder_up3_rest, ["y", "h", "x"], ["y2", "h2"]),
    ("up4_rest", _decoder_up4_rest, ["y2", "h2"], ["out"]),
]
TILED_DECODER_PROGRAM = [
    ("head", _decoder_head, ["z"], ["x0"]),
    ("up0", lambda f, x: f.up(0, x), ["x0"], ["x1u"]),
    ("up1", lambda f, x: f.up(1, x), ["x1u"], ["x2u"]),
    # up_blocks[2] as two graphs: as one, it computes wrong values at some 2048-class tile shapes (rebel-compiler
    # 0.12.1.dev163, e.g. 2528x1696 in 8 row tiles)
    ("up2_resnets", lambda f, x: f.up_resnets(2, x), ["x2u"], ["x2r"]),
    ("up2_upsample", lambda f, x, x_copy: f.up_upsample(2, x, x_copy), ["x2r", "x2u"], ["x"]),
    ("up3_branch", _decoder_up3_branch, ["x"], ["y", "h"]),
    ("up3_resnet1", _up3_resnet(1), ["y", "h"], ["x1"]),
    ("up3_resnets", _decoder_up3_rest_resnets, ["x1"], ["x2"]),
    ("up3_upsample", lambda f, x, x_copy: _upsample(f.vae.decoder.up_blocks[3].upsampler, x) + f.dup[3](x_copy),
     ["x2", "x"], ["x3"]),
    ("up4_branch", lambda f, x: _resblock_branch(f.vae.decoder.up_blocks[4].resnets[0], x), ["x3"], ["y2", "h2"]),
    ("up4_resnet1", _up4_resnet(1), ["y2", "h2"], ["x4"]),
    ("up4_resnets", _decoder_up4_rest_resnets, ["x4"], ["x5"]),
    ("out", lambda f, x: f.decoder_out(x), ["x5"], ["out"]),
]  # fmt: skip
DECODER_TILED_STEPS = {name for name, *_ in TILED_DECODER_PROGRAM[2:]}  # (up_blocks[0] runs whole: see TILE_HALO)
TILE_HALO = 8  # latent rows; the receptive radius of the tiled steps is about 6.7


def _encoder_e3(f, y, h, x_copy):
    down_block, x = f.vae.encoder.down_blocks[1], y + h
    for block in down_block.resnets[1:]:
        x = _resblock(block, x)
    return down_block.downsampler.resample(x) + f.avg[1](x_copy)


def _encoder_e4(f, x):
    encoder = f.vae.encoder
    for k in range(2, len(encoder.down_blocks)):
        x = f.down(k, x)
    x = f.mid(encoder.mid_block, x)
    x = _conv(f.vae.quant_conv, _conv(encoder.conv_out, F.silu(_rms(encoder.norm_out, x))))
    return x[:, : f.vae.config.z_dim]


# The encoder takes the image as RGBA (4 channels) and returns the posterior mean. Its first three steps are local and
# may run in row tiles planned in units of 16 pixel rows with a 2-unit halo (receptive radius about 20 pixels).
ENCODER_PROGRAM = [
    ("in_down0", lambda f, x: f.down(0, _conv(f.vae.encoder.conv_in, x)), ["x"], ["x1"]),
    ("down1_branch", lambda f, x: _resblock_branch(f.vae.encoder.down_blocks[1].resnets[0], x), ["x1"], ["y", "h"]),
    ("down1_rest", _encoder_e3, ["y", "h", "x1"], ["x3"]),
    ("down23_mid_out", _encoder_e4, ["x3"], ["z"]),
]
ENCODER_TILED_STEPS = {"in_down0", "down1_branch", "down1_rest"}
ENCODER_TILE_UNIT, ENCODER_TILE_HALO = 16, 2


def tile_plan(rows: int, n: int, halo: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Row windows of n tiles over `rows` rows: the common window height and `(start, keep_from, keep_to)` per tile."""
    if n <= 1:
        return rows, [(0, 0, rows)]
    height = min(rows, -(-rows // n) + 2 * halo)
    plan = []
    for i in range(n):
        begin, end = rows * i // n, rows * (i + 1) // n
        start = min(max(begin - halo, 0), rows - height)
        plan.append((start, begin - start, end - start))
    return height, plan


def run_program(program, graphs, env, tiled=(), n_tiles=1, halo=0, unit=1, base=None):
    """Runs a chain program over `env` (name → tensor). Steps in `tiled` run in `n_tiles` row tiles planned in units of
    `unit` rows of `env[base]`; every tensor is sliced at its own scale and the outputs of the last tiled step are
    stitched back (each tile keeps its own rows)."""
    rows = env[base].shape[2] // unit if n_tiles > 1 else 0
    height, plan = tile_plan(rows, n_tiles, halo) if n_tiles > 1 else (0, [])
    tiles = last_outputs = None
    for name, _, inputs, outputs in program:
        if tiles is not None and name not in tiled:
            _stitch(env, tiles, last_outputs, height)
            tiles = None
        if name in tiled and n_tiles > 1:
            if tiles is None:
                tiles = []
                for start, keep_from, keep_to in plan:
                    window = {}
                    for key in inputs:
                        scale = env[key].shape[2] // rows
                        window[key] = env[key][:, :, start * scale : (start + height) * scale].contiguous()
                    tiles.append((window, keep_from, keep_to))
            for window, _, _ in tiles:
                result = graphs[name](*[window[key] for key in inputs])
                window.update(zip(outputs, result if isinstance(result, list | tuple) else (result,), strict=True))
            last_outputs = outputs
        else:
            result = graphs[name](*[env[key] for key in inputs])
            env.update(zip(outputs, result if isinstance(result, list | tuple) else (result,), strict=True))
    if tiles is not None:
        _stitch(env, tiles, last_outputs, height)
    return env


def _stitch(env, tiles, keys, height):
    for key in keys:
        scale = tiles[0][0][key].shape[2] // height
        env[key] = torch.cat([window[key][:, :, scale * a : scale * b] for window, a, b in tiles], dim=2)


# --------------------------------------------------------------------------------------------------------- runtime
class _LazyRuntime:
    """A graph whose runtime is created on first use and can be released (see `RBLNAutoencoderKLQwenImage21`).

    A second runtime created from an `RBLNCompiledModel` whose first runtime was destroyed crashes (segmentation fault,
    rebel-compiler 0.12.1.dev163), so after the first one, runtimes are created from the saved file (`path`)."""

    def __init__(self, compiled_model: rebel.RBLNCompiledModel, **runtime_kwargs):
        self.compiled_model, self.runtime_kwargs, self.runtime = compiled_model, runtime_kwargs, None
        self.path = None

    def open(self):
        if self.runtime is None:
            if self.compiled_model is None and self.path is None:
                raise RuntimeError("A released VAE graph can only be loaded again from a saved model.")
            source = self.compiled_model if self.compiled_model is not None else str(self.path)
            self.runtime = rebel.Runtime(source, tensor_type="pt", **self.runtime_kwargs)
            if self.path is not None:
                self.compiled_model = None

    def close(self):
        self.runtime = None

    def __call__(self, *args):
        self.open()
        return self.runtime.run(*args)


class _LatentDistribution:
    """The posterior as the pipeline reads it (`mode()`; the mean is all the graphs compute)."""

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError("Only the posterior mean is computed; use `mode()` (`sample_mode='argmax'`).")


def _graph_name(kind: str, size: tuple[int, int], step: str) -> str:
    return f"{kind}_{size[0]}x{size[1]}_{step}"


class RBLNAutoencoderKLQwenImage21(RBLNModel):
    """
    RBLN implementation of AutoencoderKLQwenImage21, the VAE of Qwen-Image-2.1 pipelines.

    The original is a video VAE; an image is one frame, so the decoder and encoder run here as 2D graphs. Each image size
    of `sample_sizes` (decoder) and `encoder_sample_sizes` (encoder) is a chain of graphs compiled for that exact size:
    the decoder's mid-block attention and the encoder's need the whole image. At the 2048 class, the decoder runs the
    part after its mid block in row tiles (exact, see `decoder_row_tiles`).

    On RBLN-CR13 a VAE graph keeps its activations on the first chiplet of the device, so a 2048-class decoder chain
    takes about 20 GiB of it. The chains of one size are therefore loaded on first use and those of the least recently
    used sizes are released (one decoder size and two encoder sizes stay loaded). With `device=[a, b]` the decoder
    runs on device `a` and the encoder on device `b`.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = AutoencoderKLQwenImage21
    _rbln_config_class = RBLNAutoencoderKLQwenImage21Config
    MAX_LOADED = {"decoder": 1, "encoder": 2}

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.graphs = {}
        if self.model:  # (no runtimes when loaded with `create_runtimes=False`)
            names = [cfg.compiled_model_name for cfg in self.rbln_config.compile_cfgs]
            self.graphs = dict(zip(names, self.model, strict=True))
            for name, graph in self.graphs.items():
                path = Path(self.model_save_dir) / self.subfolder / f"{name}.rbln"
                graph.path = path if path.exists() else None
        self.verify_outputs = self.rbln_config.verify_outputs
        self._loaded = {"decoder": [], "encoder": []}  # sizes with open runtimes, least recently used first

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config sets the image sizes.
        return rbln_config

    @staticmethod
    def _chain_for(kind: str, size: tuple[int, int], tiles: int):
        """(kind, size, program, tiled steps, tiles, halo, unit, base) of a chain."""
        if kind == "encoder":
            return (kind, size, ENCODER_PROGRAM, ENCODER_TILED_STEPS, tiles, ENCODER_TILE_HALO, ENCODER_TILE_UNIT, "x")
        if tiles == 1:
            return (kind, size, DECODER_PROGRAM, set(), 1, 0, 1, "z")
        return (kind, size, TILED_DECODER_PROGRAM, DECODER_TILED_STEPS, tiles, TILE_HALO, 1, "z")

    @classmethod
    def _programs(cls, rbln_config: RBLNAutoencoderKLQwenImage21Config):
        """The chains of every size (see `_chain_for`)."""
        return [
            cls._chain_for("decoder", size, rbln_config.tiles("decoder", *size)) for size in rbln_config.sample_sizes
        ] + [
            cls._chain_for("encoder", size, rbln_config.tiles("encoder", *size))
            for size in rbln_config.encoder_sample_sizes
        ]

    @staticmethod
    def _modules(model: AutoencoderKLQwenImage21) -> dict[str, _VAE2D]:
        # The decoder runs in bfloat16 (PSNR 50.9 dB against the float32 original); the encoder in float32, which the
        # compiler lowers to dlfloat16 (in bfloat16 the condition latents were 1.5 % off, against 0.5-0.9 % for the
        # bfloat16 original on CPU, which shows in edits).
        return {
            "decoder": _VAE2D(copy.deepcopy(model).to(torch.bfloat16)).eval(),
            "encoder": _VAE2D(_split_encoder_convs(copy.deepcopy(model).to(torch.float32))).eval(),
        }

    @staticmethod
    def _dtype(kind: str) -> torch.dtype:
        return torch.bfloat16 if kind == "decoder" else torch.float32

    @classmethod
    def _chain_input_info(cls, modules, kind, size, program, tiled, tiles, halo, unit):
        """Input shapes of every step at a size: the program runs on a small probe on CPU to get each tensor's channels
        and scale, and the shapes are scaled up (tiled steps take a window of the common tile height)."""
        dtype = cls._dtype(kind)
        probe = 2 * unit
        start = {"z": torch.zeros(1, 64, 2, 2, dtype=dtype)} if kind == "decoder" else {}
        if kind == "encoder":
            start = {"x": torch.zeros(1, 4, probe, probe, dtype=dtype)}
        steps = {name: _Step(modules[kind], fn) for name, fn, _, _ in program}
        with torch.no_grad():
            env = run_program(program, steps, dict(start))
        height, width = (size[0] // 16, size[1] // 16) if kind == "decoder" else size
        rows = height // unit
        tile_rows = tile_plan(rows, tiles, halo)[0]
        info = {}
        for name, _, inputs, _ in program:
            step_rows = tile_rows if name in tiled and tiles > 1 else rows
            info[name] = [
                (
                    key,
                    [
                        1,
                        env[key].shape[1],
                        step_rows * env[key].shape[2] * unit // probe,
                        width * env[key].shape[3] // probe,
                    ],
                    str(dtype).split(".")[-1],
                )
                for key in inputs
            ]
        return info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLQwenImage21Config,
    ) -> RBLNAutoencoderKLQwenImage21Config:
        modules = cls._modules(model)
        rbln_config.set_compile_cfgs(
            [cfg for chain in cls._programs(rbln_config) for cfg in cls._chain_compile_cfgs(modules, chain)]
        )
        return rbln_config

    @classmethod
    def _chain_compile_cfgs(cls, modules, chain) -> list[RBLNCompileConfig]:
        kind, size, program, tiled, tiles, halo, unit, _ = chain
        info = cls._chain_input_info(modules, kind, size, program, tiled, tiles, halo, unit)
        return [
            RBLNCompileConfig(compiled_model_name=_graph_name(kind, size, name), input_info=info[name])
            for name, *_ in program
        ]

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        modules = cls._modules(model)
        functions = {
            (kind, name): fn
            for kind, program in (("decoder", DECODER_PROGRAM + TILED_DECODER_PROGRAM), ("encoder", ENCODER_PROGRAM))
            for name, fn, _, _ in program
        }
        compiled_models, compile_cfgs = {}, []
        for chain in cls._programs(rbln_config):
            kind, size, *_, tiles, _, _, _ = chain
            # A chain whose graphs do not compile at a size (shape-specific compiler errors, e.g. the encoder at
            # 1248x832) or exceed `max_chain_gib` of chiplet 0 runs in more row tiles (the result is the same).
            candidates = [tiles] + [n for n in (2, 3, 4, 6, 8, 12, 16) if n > tiles]
            for more in candidates:
                chain = cls._chain_for(kind, size, more)
                cfgs = cls._chain_compile_cfgs(modules, chain)
                try:
                    chain_models = {
                        cfg.compiled_model_name: cls.compile(
                            _Step(modules[kind], functions[(kind, cfg.compiled_model_name.split("_", 2)[2])]).eval(),
                            rbln_compile_config=cfg,
                            create_runtimes=False,  # (runtimes are created on first use)
                            device=rbln_config.device,
                        )
                        for cfg in cfgs
                    }
                except rebel.core.exception.RBLNCompileError:
                    if more == candidates[-1]:
                        raise
                    logger.warning(f"The VAE {kind} at {size[0]}x{size[1]} does not compile in {more} row tile(s).")
                    continue
                gib = (
                    sum(sum(v[0][0] for v in m.get_alloc_per_chiplet_by_key().values()) for m in chain_models.values())
                    / 2**30
                )
                if gib > rbln_config.max_chain_gib and more != candidates[-1] and len(chain[3]) > 0:
                    logger.warning(
                        f"The VAE {kind} at {size[0]}x{size[1]} in {more} row tile(s) takes {gib:.1f} GiB of chiplet 0 "
                        f"(`max_chain_gib={rbln_config.max_chain_gib}`); trying more tiles."
                    )
                    continue
                if more != tiles:
                    rbln_config.row_tiles[f"{kind}_{size[0]}x{size[1]}"] = more
                compiled_models.update(chain_models)
                compile_cfgs += cfgs
                break
        rbln_config.set_compile_cfgs(compile_cfgs)
        return compiled_models

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> list[_LazyRuntime]:
        def device(name):  # a list of two devices places the decoder on the first and the encoder on the second
            if isinstance(rbln_config.device, list | tuple):
                return rbln_config.device[0] if name.startswith("decoder") else rbln_config.device[-1]
            return rbln_config.device_map[name]

        return [
            _LazyRuntime(
                compiled_model,
                device=device(cfg.compiled_model_name),
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for cfg, compiled_model in zip(rbln_config.compile_cfgs, compiled_models, strict=True)
        ]

    # ------------------------------------------------------------------------------------------------ inference
    def _run_graph(self, graph: _LazyRuntime, *args):
        output = graph(*args)
        if not self.verify_outputs:
            return output
        as_list = lambda o: o if isinstance(o, list | tuple) else [o]  # noqa: E731
        for _ in range(6):  # until two runs agree
            again = graph(*args)
            if all(torch.equal(a, b) for a, b in zip(as_list(again), as_list(output), strict=True)):
                return output
            output = again
        raise RuntimeError("A VAE graph gave a different output on every run.")

    def _chain(self, kind: str, size: tuple[int, int]):
        chain = next((c for c in self._programs(self.rbln_config) if c[0] == kind and c[1] == size), None)
        if chain is None:
            option = "sample_sizes" if kind == "decoder" else "encoder_sample_sizes"
            sizes = getattr(self.rbln_config, option)
            raise ValueError(f"The VAE {kind} was not compiled for {size[0]}x{size[1]} (`{option}={sizes}`).")
        loaded = self._loaded[kind]
        if size in loaded:
            loaded.remove(size)
        while len(loaded) >= self.MAX_LOADED[kind]:  # release the least recently used size first
            for name, *_ in self._programs_by_size(kind, loaded.pop(0)):
                self.graphs[name].close()
            gc.collect()  # (the device memory is given back when the runtimes are destroyed)
        loaded.append(size)
        _, _, program, tiled, tiles, halo, unit, base = chain
        graphs = {name: partial(self._run_graph, self.graphs[_graph_name(kind, size, name)]) for name, *_ in program}
        return program, graphs, {"tiled": tiled, "n_tiles": tiles, "halo": halo, "unit": unit, "base": base}

    def _programs_by_size(self, kind, size):
        program = next(c[2] for c in self._programs(self.rbln_config) if c[0] == kind and c[1] == size)
        return [(_graph_name(kind, size, name), *rest) for name, *rest in program]

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        """
        Decodes latents into an image.

        Args:
            z (torch.Tensor): Denormalized latents of shape `(batch_size, 64, 1, height / 16, width / 16)`.
            return_dict (bool): Whether to return a `DecoderOutput` instead of a plain tuple.

        Returns:
            The image of shape `(batch_size, 4, 1, height, width)` in `[-1, 1]`.
        """
        images = []
        for sample in z:  # the graphs take one image
            latent = sample[:, 0][None].to(torch.bfloat16).contiguous()
            program, graphs, tiling = self._chain("decoder", (latent.shape[2] * 16, latent.shape[3] * 16))
            images.append(run_program(program, graphs, {"z": latent}, **tiling)["out"].float())
        image = torch.cat(images).unsqueeze(2)
        return DecoderOutput(sample=image) if return_dict else (image,)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        """
        Encodes condition images into the posterior mean.

        Args:
            x (torch.Tensor): Images of shape `(batch_size, 4, 1, height, width)` in `[-1, 1]` (RGBA).
            return_dict (bool): Whether to return an `AutoencoderKLOutput` instead of a plain tuple.

        Returns:
            The posterior, of which only `mode()` (the mean) is computed.
        """
        means = []
        for sample in x:
            image = sample[:, 0][None].float().contiguous()
            program, graphs, tiling = self._chain("encoder", (image.shape[2], image.shape[3]))
            means.append(run_program(program, graphs, {"x": image}, **tiling)["z"].float())
        posterior = _LatentDistribution(torch.cat(means).unsqueeze(2))
        return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)


__all__ = [
    "RBLNAutoencoderKLQwenImage21",
]
