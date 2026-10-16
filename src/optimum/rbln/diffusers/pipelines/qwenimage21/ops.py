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

"""Op replacements to match reference accuracy on RBLN.

On CR13 / rebel-compiler 0.11.3.dev606, SiLU is lowered to the same kernel whether written as F.silu or x*sigmoid(x),
which gives a per-token relative error of 6.6 % on the gate value distribution of the Qwen3-VL MLP (CPU bf16: 0.3 %).
Writing it out as x / (1 + exp(-x)) brings it down to 0.69 %, close to CPU.
"""

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Same as transformers' rotate_half (RoPE)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)


def patch_silu(module: nn.Module) -> int:
    """Replace nn.SiLU and transformers SiLU activation modules inside `module` with the exact SiLU. Returns the number replaced."""
    n = 0
    for name, child in list(module.named_children()):
        cls = type(child).__name__
        if isinstance(child, nn.SiLU) or cls in ("SiLUActivation",):
            setattr(module, name, SiLU())
            n += 1
        else:
            n += patch_silu(child)
    return n


def split_timestep(t: torch.Tensor) -> torch.Tensor:
    """t ≥ 0 (fp32) → [..., 3] bf16-valued pieces h1 + h2 + h3 = t (exact: 3 × 8 mantissa bits cover fp32's 24), as float32
    graph input. Truncation (not rounding) keeps every piece ≥ 0, which the table lookup needs."""

    def trunc(x):  # largest bf16 value ≤ x (x ≥ 0): drop the low 16 bits of the fp32 encoding
        return (x.contiguous().view(torch.int32) & -65536).view(torch.float32)

    t = t.float()
    h1 = trunc(t)
    h2 = trunc(t - h1)
    return torch.stack([h1, h2, t - h1 - h2], dim=-1)


class ExactTimestepProj(nn.Module):
    """The timestep sinusoid ([cos, sin] of 1000·t·freq) of the original, computed on the NPU without phase arithmetic.

    Elementwise ops on CR13 are bf16/dlfloat16 precision, so phases reaching 1000 rad cannot be computed arithmetically
    (the original formulation gives 10-17 % relative error). Instead:
      1) a bf16 value t is located exactly with comparisons and power-of-two multiplications (exponent j: t ∈ [2^-j, 2^-j+1),
         mantissa m: 0..127), giving a one-hot row of a table [n_exp·128 + 1, dim] precomputed in fp64;
      2) an fp32 timestep is passed as three bf16 pieces t = h1 + h2 + h3 (`split_timestep`); each piece is looked up and
         the angles are added with cos(a+b) = cos a cos b - sin a sin b, sin(a+b) = sin a cos b + cos a sin b.
    Rounding t itself to bf16 (978.5 → 976.6) changes the DiT output by 4.5 % at the first steps; the pieces keep it exact.
    Pieces below 2^-(n_exp-1) are treated as 0 (phase error ≤ 1000·2^-23 ≈ 1.2e-4 rad, below the dlfloat16 rounding of
    the table values on the NPU).
    """

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0, n_exp: int = 24):
        super().__init__()
        import math

        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        # rows: (j, m) → t = (128 + m) · 2^(-j-7), j = 0..n_exp-1; the last row is t = 0
        j = torch.arange(n_exp, dtype=torch.float64)[:, None]
        m = torch.arange(128, dtype=torch.float64)[None, :]
        t = ((128 + m) * torch.pow(2.0, -j - 7)).reshape(-1)
        t = torch.cat([t, torch.zeros(1, dtype=torch.float64)])
        args = (t * time_factor)[:, None] * freqs.double()[None]
        self.register_buffer("table", torch.cat([torch.cos(args), torch.sin(args)], dim=-1).float(), persistent=False)
        bf = torch.bfloat16
        self.register_buffer("thr", torch.pow(2.0, -torch.arange(n_exp, dtype=torch.float64)).to(bf), persistent=False)
        self.register_buffer(
            "scale", torch.pow(2.0, torch.arange(n_exp, dtype=torch.float64) + 7).to(bf), persistent=False
        )
        self.register_buffer("mant", torch.arange(128, dtype=torch.float64).to(bf), persistent=False)
        self.half = half

    def _lookup(self, t: torch.Tensor) -> torch.Tensor:  # t [N] bf16 values → [N, dim]
        dt = torch.bfloat16
        t = t.to(dt)
        s = (t[:, None] >= self.thr[None]).to(dt)  # [N, n_exp], 1 for larger j (smaller threshold)
        s_prev = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], dim=1)
        onehot_e = s - s_prev  # 1 only at the first j that becomes 1
        scale = (onehot_e * self.scale[None]).sum(dim=1)  # 2^(j+7), or 0 if none matches
        m = t * scale - 128  # 0..127 (exact)
        onehot_m = (m[:, None] == self.mant[None]).to(dt)  # [N, 128]
        onehot = (onehot_e[:, :, None] * onehot_m[:, None, :]).reshape(t.shape[0], -1)
        zero = 1 - onehot_e.sum(dim=1, keepdim=True)  # 1 if t < 2^-(n_exp-1)
        onehot = torch.cat([onehot, zero], dim=1)
        return onehot.to(self.table.dtype) @ self.table

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] (bf16 values) or [B, 3] (`split_timestep` pieces) → [B, dim]."""
        if t.dim() == 1:
            return self._lookup(t)
        B, h = t.shape[0], self.half
        cs = self._lookup(t.reshape(-1)).reshape(B, 3, 2 * h)
        c, s = cs[:, 0, :h], cs[:, 0, h:]
        for i in (1, 2):
            ci, si = cs[:, i, :h], cs[:, i, h:]
            c, s = c * ci - s * si, s * ci + c * si
        return torch.cat([c, s], dim=-1)
