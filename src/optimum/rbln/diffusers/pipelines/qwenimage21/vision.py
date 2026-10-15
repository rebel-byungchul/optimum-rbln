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

"""Qwen3-VL vision encoder (condition images in edit mode) — static graph per single image (RBLN).

The original Qwen3VLVisionModel concatenates multiple images and runs attention per image using cu_seqlens.
Here each image is processed separately with a patch bucket (trailing padding, key mask). The result is identical.
Inside the graph: patch embed (same computation as a Linear), position embedding gather + weighted sum (interpolation indices and weights are
metadata determined by the grid size), 2D axial RoPE table gather, 27 blocks, 3 deepstack mergers, final merger.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from .ops import rotate_half


NEG = -1e9


class GeluTanhExp(nn.Module):
    """Same function as gelu(approximate="tanh"): 0.5x(1+tanh z) = x / (1 + exp(-2z)). On the NPU this form is more accurate
    than the tanh kernel (over 4 vision blocks: 1.08 % → 0.71 % relative error)."""

    def forward(self, x):
        z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        return x / (1 + torch.exp(-2 * z))


class VisionStatic(nn.Module):
    """inputs: patches [Np, C·T·P·P], interp_idx [Np, 4] int64, interp_w [Np, 4], pos [2, Np] int64, n_valid [1] int32
    outputs: merged [Np/4, 4096], deepstack [n_ds, Np/4, 4096]"""

    MAX_POS = 1024  # rows of the 2D RoPE table (patch grid side)

    def __init__(self, visual):
        super().__init__()
        self.v = visual
        for b in visual.blocks:
            b.mlp.act_fn = GeluTanhExp()
        pe = visual.patch_embed
        w = pe.proj.weight  # [D, C, T, P, P]
        self.patch_lin = nn.Linear(w[0].numel(), w.shape[0], bias=True)
        self.patch_lin.weight.data = w.reshape(w.shape[0], -1).clone()
        self.patch_lin.bias.data = pe.proj.bias.data.clone()
        rot = visual.rotary_pos_emb
        inv = rot.inv_freq.float()
        freqs = torch.arange(self.MAX_POS, dtype=torch.float32)[:, None] * inv[None]
        sc = float(rot.attention_scaling)
        self.rcos = nn.Embedding.from_pretrained(freqs.cos() * sc, freeze=True)
        self.rsin = nn.Embedding.from_pretrained(freqs.sin() * sc, freeze=True)

    def _rope(self, pos):  # pos [2, Np] (h, w) → cos/sin [Np, head_dim]
        c = torch.cat([self.rcos(pos[0]), self.rcos(pos[1])], -1)
        s = torch.cat([self.rsin(pos[0]), self.rsin(pos[1])], -1)
        return torch.cat([c, c], -1), torch.cat([s, s], -1)

    def forward(self, patches, interp_idx, interp_w, pos, n_valid):
        v = self.v
        Np = patches.shape[0]
        x = self.patch_lin(patches)
        pe = (v.pos_embed(interp_idx) * interp_w[:, :, None]).sum(1)
        x = x + pe.to(x.dtype)
        cos, sin = self._rope(pos)
        key_bias = (1 - (torch.arange(Np, dtype=torch.int32) < n_valid).float()) * NEG
        key_bias = key_bias[None, None, None, :]
        ds = []
        for li, blk in enumerate(v.blocks):
            at = blk.attn
            h = blk.norm1(x)
            qkv = at.qkv(h).reshape(Np, 3, at.num_heads, -1).permute(1, 0, 2, 3)
            q, k, vv = qkv[0], qkv[1], qkv[2]  # [Np, H, d]
            c, s = cos[:, None].float(), sin[:, None].float()
            q = (q.float() * c + rotate_half(q.float()) * s).to(vv.dtype)
            k = (k.float() * c + rotate_half(k.float()) * s).to(vv.dtype)
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], vv.transpose(0, 1)[None], attn_mask=key_bias
            )
            o = o[0].transpose(0, 1).reshape(Np, -1)
            x = x + at.proj(o)
            x = x + blk.mlp(blk.norm2(x))
            if li in v.deepstack_visual_indexes:
                ds.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(li)](x))
        return v.merger(x), torch.stack(ds, 0)


def vision_metadata(visual, grid_thw_one: torch.Tensor, Np_bucket: int):
    """Build interpolation indices, weights and RoPE positions for one image (grid_thw [1,3]), padded to the bucket length."""
    idx, w = get_vision_interpolation_indices_and_weights(
        grid_thw_one,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    pos = get_vision_position_ids(grid_thw_one, visual.spatial_merge_size).T  # [2, n]
    n = idx.shape[0]
    if n > Np_bucket:
        raise ValueError(f"patches {n} > bucket {Np_bucket}")
    I = torch.zeros(Np_bucket, idx.shape[1], dtype=torch.long)
    I[:n] = idx
    Wt = torch.zeros(Np_bucket, w.shape[1])
    Wt[:n] = w.float()
    P = torch.zeros(2, Np_bucket, dtype=torch.long)
    P[:, :n] = pos.long()
    return I, Wt, P, n
