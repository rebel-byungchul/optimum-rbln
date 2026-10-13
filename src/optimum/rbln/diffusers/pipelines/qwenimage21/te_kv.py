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

"""Qwen-Image-2.1 text encoder (Qwen3-VL 8B language model) — chunked prefill + device-resident KV cache (RBLN).

The original pipeline uses `text_encoder(...).hidden_states[-1]` and disables the final RMSNorm with a norm hook.
So the value needed is the **output of the 36th decoder layer (before the final norm)**. The language model is fully causal, including image tokens, so
splitting the sequence into chunks and processing them in order with `paged_causal_attn_prefill` (eager, causal) matches the original.

Done inside the graph:
  - token embedding gather (input_ids, int64)
  - (mm graph) replace image token positions with the vision encoder output; add deepstack features after layers 0-2
  - mRoPE: gather per-axis (t,h,w) cos/sin tables by position, then select channels by the interleaved layout (mrope_section)
  - SiLU is ops.silu(x/(1+exp(-x))) — works around the NPU SiLU kernel precision issue (10-13)
The host only builds integer metadata (positions, cache positions, valid mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ops import patch_silu


@dataclass
class TEConfig:
    cap: int = 32256  # eager cache length (one block); int16 position limit < 32768
    kv_heads: int = 8
    heads: int = 32
    head_dim: int = 128
    rope_rows: int = 32768  # number of mRoPE table rows (max position)


class MRopeGather(nn.Module):
    """Build the same cos/sin as Qwen3VLTextRotaryEmbedding by gathering at positions [3, C] → [C, 128]."""

    def __init__(self, rotary, rows: int):
        super().__init__()
        inv = rotary.inv_freq.float()  # [64]
        pos = torch.arange(rows, dtype=torch.float32)[:, None]
        freqs = pos @ inv[None]  # original: inv_freq @ position (fp32)
        sc = float(rotary.attention_scaling)
        self.cos = nn.Embedding.from_pretrained(freqs.cos() * sc, freeze=True)
        self.sin = nn.Embedding.from_pretrained(freqs.sin() * sc, freeze=True)
        sel = torch.zeros(3, inv.numel())  # which axis position each channel uses (original recomposition_frequencies)
        sel[0] = 1
        for dim, offset in enumerate((1, 2), start=1):
            length = rotary.mrope_section[dim] * 3
            sel[0, offset:length:3] = 0
            sel[dim, offset:length:3] = 1
        self.register_buffer("sel", sel, persistent=False)

    def forward(self, pos):  # pos [3, C] int64
        c = sum(self.cos(pos[a]) * self.sel[a] for a in range(3))
        s = sum(self.sin(pos[a]) * self.sel[a] for a in range(3))
        return torch.cat([c, c], -1), torch.cat([s, s], -1)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TEKVShared(nn.Module):
    """Auxiliary tensors shared by the graphs (mRoPE table, scale). All graphs reference this object so they fall into the same weight group."""

    def __init__(self, lm, cfg: TEConfig, precise_silu=True):
        super().__init__()
        if precise_silu:
            patch_silu(lm)
        self.lm = lm
        self.rope = MRopeGather(lm.rotary_emb, cfg.rope_rows)
        self.scale = nn.Parameter(torch.tensor(cfg.head_dim**-0.5), requires_grad=False)


class TEKVSegment(nn.Module):
    """One graph for language model layers [l0, l1). kind: "txt" (text only) | "mm" (includes image tokens, deepstack).

    inputs: first: input_ids [C] int64 (+ mm: vis [C, 4096], is_vis [C], ds [n_ds, C, 4096]) / otherwise hidden [C, 4096]
            pos [3, C] int64, cache_pos [1, C] int32, block_tables [1] int16, mask [1, CAP], kc, vc per layer
    output: hidden [C, 4096] (for the last graph, the output of the last layer, before norm)
    """

    def __init__(self, shared, l0: int, l1: int, kind: str, cfg: TEConfig, n_deepstack: int = 3):
        super().__init__()
        if not isinstance(shared, TEKVShared):
            shared = TEKVShared(shared, cfg)
        self.sh = shared
        self.l0, self.l1, self.kind, self.cfg = l0, l1, kind, cfg
        self.layers = list(shared.lm.layers[l0:l1])
        self.first = l0 == 0
        self.n_ds = n_deepstack if kind == "mm" else 0

    @property
    def lm(self):
        return self.sh.lm

    @property
    def rope(self):
        return self.sh.rope

    @property
    def scale(self):
        return self.sh.scale

    def _attn(self, at, x, cos, sin, kc, vc, cache_pos, bt, mask):
        cfg = self.cfg
        C = x.shape[1]
        q = at.q_norm(at.q_proj(x).view(1, C, cfg.heads, cfg.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(1, C, cfg.kv_heads, cfg.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(1, C, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q, k = q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        g = cfg.heads // cfg.kv_heads
        o = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q.reshape(1, cfg.kv_heads, g, C, cfg.head_dim),
            k=k.unsqueeze(2),
            v=v.unsqueeze(2),
            kcache=kc.unsqueeze(2),
            vcache=vc.unsqueeze(2),
            seq=cache_pos[:, :1],
            scale=self.scale,
            block_table=bt,
            block_size=cfg.cap,
            is_bidirectional=False,
            mask=mask,
        )
        o = o.view(1, cfg.heads, C, cfg.head_dim).transpose(1, 2).reshape(1, C, -1)
        return at.o_proj(o)

    def forward(self, x, *args):
        args = list(args)
        if self.first:
            ids = x
            h = self.lm.embed_tokens(ids)
            if self.kind == "mm":
                vis, is_vis = args.pop(0), args.pop(0)
                m = is_vis[:, None]
                h = h * (1 - m) + vis * m  # same as the original masked_scatter (m ∈ {0,1})
        else:
            h = x
        ds = args.pop(0) if (self.kind == "mm" and self.l0 < self.n_ds) else None
        pos, cache_pos, bt, mask = args[:4]
        caches = args[4:]
        cos, sin = self.rope(pos)
        h = h[None]
        for i, layer in enumerate(self.layers):
            li = self.l0 + i
            kc, vc = caches[2 * i], caches[2 * i + 1]
            r = h
            h = layer.input_layernorm(h)
            h = r + self._attn(layer.self_attn, h, cos, sin, kc, vc, cache_pos, bt, mask)
            r = h
            h = r + layer.mlp(layer.post_attention_layernorm(h))
            if ds is not None and li < self.n_ds:
                h = h + ds[li][None]  # deepstack: only visual token rows are nonzero (the rest are 0)
        return h[0]
