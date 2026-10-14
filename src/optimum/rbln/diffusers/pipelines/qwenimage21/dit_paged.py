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

"""Qwen-Image-2.1 DiT for 2048-class outputs next to long prefixes ("xxl"): per-layer weight graphs + shared attention.

Why (CR13, rebel-compiler dev606, measured 10-14):
- paged flash attention is wrong across partitions for chunks of 16384 tokens or more, and the eager cache must stay below
  32768 tokens, so a 2048² target (16384 tokens) or an output_resolution-2048 condition image next to a long prefix fits
  neither the flash nor the eager configuration;
- the flash op's Init scratch grows with query chunk × visible cache (1.28 GiB for 8448 × 53760) and is allocated per
  graph: one attention per layer graph puts ~3 GiB per layer on a device, which capped the cache at two 2048-class
  condition images.

Layout here: per device, one static block cache (all layers of the device, `blocks_per_layer` blocks of `part` tokens
each) and a few attention graphs without weights (one per query chunk size) that every layer shares — the block table
selects the layer's blocks, so the Init scratch exists once per device. The layer computations (norms, modulation,
q/k/v, RoPE, output projection, MLP) are per-layer weight graphs whose q/k/v and attention outputs go through the host.

A block of N tokens above one piece (the target, a large condition image) is split in two pieces A, B (≤ PIECE tokens,
one block each). Keys carry their RoPE, so the order of blocks in a block table does not change attention; per layer:
  write A          (block table ... A)        A's K/V into its block (output not used)
  B attends        (block table ... A, B)     prefix + A + B  = B's rows of the original bidirectional attention
  A attends again  (block table ... B, A)     prefix + B + A  = A's rows (rewrites the same K/V)
Text chunks (causal) and condition images up to 4352 tokens are single chunks.
"""

from __future__ import annotations

import torch
from torch import nn

from .dit_kv import DiTKVShared, rope_real


PIECE = 8448  # tokens per piece: a 2048-class block (≤ 16896 tokens) is two pieces
PART = PIECE + 512  # block size: a piece and a text chunk
BLOCKS_PER_LAYER = 12  # per layer: 2 target blocks + 10 prefix blocks (e.g. five output_resolution-2048 images + text)
CHUNKS = {
    "txt64": (64, False),
    "txt512": (512, False),
    "img": (4352, True),
    "piece": (PIECE, True),
}  # attention graphs
WEIGHT_C = {"txt": 512, "block": 2 * PIECE}  # rows of the layer graphs (txt: layer 0's text embedding only)


class PagedAttention(nn.Module):
    """Paged flash attention over the block cache, no weights. q, k, v [C, H, D] bf16 → o [C, H, D] bf16.
    seq [1, 1]: logical position of the chunk; bt [nv]: physical blocks in logical order; mask [1, nv × part]."""

    def __init__(self, C, bidir, part, heads=32, head_dim=128):
        super().__init__()
        self.C, self.bidir, self.part, self.H, self.D = C, bidir, part, heads, head_dim
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=False)

    def forward(self, q, k, v, seq, bt, mask, kc, vc):
        C, H, D = self.C, self.H, self.D

        def lay(x):
            return x.float().transpose(0, 1).reshape(1, H, 1, C, D)

        o = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
            q=lay(q),
            k=lay(k),
            v=lay(v),
            kcache=kc,
            vcache=vc,
            seq=seq,
            scale=self.scale,
            block_table=bt,
            block_size=self.part,
            is_bidirectional=self.bidir,
            mask=mask,
            partition=self.part,
        )
        return o.view(H, C, D).transpose(0, 1).to(torch.bfloat16)


class _LayerBase(nn.Module):
    def __init__(self, shared: DiTKVShared, heads=32):
        super().__init__()
        self.sh, self.H = shared, heads

    @property
    def m(self):
        return self.sh.m

    @staticmethod
    def _mod(mod):  # mod [1, 4D] from DiTPagedMod (the modulation is the same for every layer)
        s1, g1, s2, g2 = mod[:, None].chunk(4, dim=-1)
        return s1, g1.tanh(), s2, g2.tanh()

    def _qkv(self, blk, x, s1, rope_idx):
        attn, H = blk.attn, self.H
        cos, sin = self.sh.rope(rope_idx)
        h = blk.img_norm1(x) * (1 + s1)
        q = attn.norm_q(attn.to_q(h).unflatten(-1, (H, -1)))
        k = attn.norm_k(attn.to_k(h).unflatten(-1, (H, -1)))
        v = attn.to_v(h).unflatten(-1, (H, -1))
        q, k = rope_real(q.to(v.dtype), cos, sin), rope_real(k.to(v.dtype), cos, sin)
        return tuple(t[0].to(torch.bfloat16) for t in (q, k, v))  # [C, H, D]


class DiTPagedMod(_LayerBase):
    """Timestep → modulation [1, 4D] and output-norm scale [1, D] (inputs of the layer graphs)."""

    def forward(self, t):
        m = self.m
        temb = m.time_text_embed.timestep_embedder(self.sh.tproj(t).to(t.dtype))
        return m.modulation(temb), m.norm_out.linear(m.norm_out.silu(temb))


class DiTPagedPre(_LayerBase):
    """q/k/v of layer `layer` from its input (layer 0: text rows through txt_in, latents through img_in; otherwise the
    hidden state). inputs: x, mod [1, 4D], rope_idx [3, C] → hidden [C, 4096] bf16, q, k, v [C, H, D] bf16."""

    def __init__(self, shared, layer: int, txt: bool):
        super().__init__(shared)
        self.layer, self.txt = layer, txt
        self._blk = [shared.m.transformer_blocks[layer]]  # a list: the parameters stay under `sh`

    def forward(self, x, mod, rope_idx):
        if self.layer == 0:
            x = self.m.txt_in(x) if self.txt else self.m.img_in(x)
        x = x.float()[None]
        s1, *_ = self._mod(mod)
        return (x[0].to(torch.bfloat16), *self._qkv(self._blk[0], x, s1, rope_idx))


class DiTPagedMid(_LayerBase):
    """The rest of layer `layer` (output projection, gated residuals, MLP), then the q/k/v of the next layer (`nxt`).
    inputs: h [C, 4096] bf16, o [C, H, D] bf16, mod, rope_idx → hidden bf16 (+ q, k, v when nxt)."""

    def __init__(self, shared, layer: int, nxt: bool):
        super().__init__(shared)
        self.layer, self.nxt = layer, nxt
        blocks = shared.m.transformer_blocks
        self._blks = [blocks[layer], blocks[layer + 1] if nxt else None]

    def forward(self, h, o, mod, rope_idx):
        blk = self._blks[0]
        x = h.float()[None]
        s1, g1, s2, g2 = self._mod(mod)
        x = x + g1 * blk.attn.to_out[0](o.float().flatten(1, 2)[None])
        x = x + g2 * blk.img_mlp(blk.img_norm2(x) * (1 + s2))
        if not self.nxt:
            return x[0].to(torch.bfloat16) + rope_idx[:1, :1].to(torch.bfloat16) * 0  # (rope_idx: same inputs)
        return (x[0].to(torch.bfloat16), *self._qkv(self._blks[1], x, s1, rope_idx))


class DiTPagedPost(_LayerBase):
    """Last layer of the target: the rest of the layer, then norm_out / proj_out → noise_pred [C, 64] fp32."""

    def __init__(self, shared, layer: int):
        super().__init__(shared)
        self._blk = [shared.m.transformer_blocks[layer]]

    def forward(self, h, o, mod, nscale):
        blk, m = self._blk[0], self.m
        x = h.float()[None]
        _, g1, s2, g2 = self._mod(mod)
        x = x + g1 * blk.attn.to_out[0](o.float().flatten(1, 2)[None])
        x = x + g2 * blk.img_mlp(blk.img_norm2(x) * (1 + s2))
        return m.proj_out(m.norm_out.norm(x) * (1 + nscale[:, None]))[0]
