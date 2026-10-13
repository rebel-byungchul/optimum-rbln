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
which gives a per-token relative error of 6.6 % on the gate value distribution of the Qwen3-VL MLP (CPU bf16: 0.3 %; measured 2026-10-13,
scripts/te_mlp_probe.py). Writing it out as x / (1 + exp(-x)) brings it down to 0.69 %, close to CPU.
"""

import torch
from torch import nn


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


class ExactTimestepProj(nn.Module):
    """Compute the timestep sinusoid ([cos, sin] of 1000·t·freq) accurately on the NPU.

    Elementwise ops on CR13 are effectively bf16 precision (10-13, scripts/precision_probe2.py), so phases reaching 1000 rad
    cannot be matched arithmetically (the original formulation gives 10-17 % relative error). Instead:
      1) using the fact that the input t is a bf16 value, build one-hot vectors via comparisons and power-of-two
         multiplications (all exact in bf16) for the exponent j (t ∈ [2^-j-1, 2^-j)) and the mantissa m (0..127), and
      2) read a table [n_exp·128 + 1, dim], precomputed in fp64 and stored as bf16, with a one-hot matmul.
    t < 2^-n_exp is treated as 0 (phase ≤ 1000·2^-24 rad).
    The target is the same bf16 value as the original (computed in fp32, then cast to bf16).
    """

    def __init__(
        self,
        timestep_dim: int = 256,
        max_period: int = 10000,
        time_factor: float = 1000.0,
        n_exp: int = 24,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        import math

        half = timestep_dim // 2
        # same fp32 computation as the original QwenImage21TemporalTimesteps
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        # rows: (j, m) → t = (128 + m) · 2^(-j-7), j = 0..n_exp-1; the last row is t = 0
        j = torch.arange(n_exp, dtype=torch.float64)[:, None]
        m = torch.arange(128, dtype=torch.float64)[None, :]
        t = ((128 + m) * torch.pow(2.0, -j - 7)).reshape(-1)
        t = torch.cat([t, torch.zeros(1, dtype=torch.float64)])
        # same order as the original: t(bf16) → float32 → ×1000 → ×freqs(float32) → cos/sin(float32) → bf16
        tf = t.float() * time_factor
        args = tf[:, None] * freqs[None]
        table = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        self.register_buffer("table", table, persistent=False)  # [n_exp*128 + 1, dim]
        self.register_buffer(
            "thr", torch.pow(2.0, -torch.arange(n_exp, dtype=torch.float64)).to(dtype), persistent=False
        )
        self.register_buffer(
            "scale", torch.pow(2.0, torch.arange(n_exp, dtype=torch.float64) + 7).to(dtype), persistent=False
        )
        self.register_buffer("mant", torch.arange(128, dtype=torch.float64).to(dtype), persistent=False)
        self.n_exp = n_exp

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dt = self.table.dtype
        t = t.to(dt)
        s = (t[:, None] >= self.thr[None]).to(dt)  # [B, n_exp], 1 for larger j (smaller threshold)
        s_prev = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], dim=1)
        onehot_e = s - s_prev  # 1 only at the first j that becomes 1
        scale = (onehot_e * self.scale[None]).sum(dim=1)  # 2^(j+7), or 0 if none matches
        m = t * scale - 128  # 0..127 (exact)
        onehot_m = (m[:, None] == self.mant[None]).to(dt)  # [B, 128]
        onehot = (onehot_e[:, :, None] * onehot_m[:, None, :]).reshape(t.shape[0], -1)
        zero = 1 - onehot_e.sum(dim=1, keepdim=True)  # 1 if t < 2^-(n_exp)
        onehot = torch.cat([onehot, zero], dim=1)
        return onehot @ self.table
