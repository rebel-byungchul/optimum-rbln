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

"""Qwen-Image-2.1 DiT for long sequences: bidirectional blocks above the flash chunk limit, with two KV caches per layer.

Why (CR13, rebel-compiler dev606, measured 10-13/14): paged flash attention is wrong across partitions for chunks of 16384
tokens or more, and the eager cache must stay below 32768 tokens. A 2048² target (16384 tokens) or a condition image at
output_resolution 2048 next to a long prefix therefore fits neither. A paged attention op can also be called only once per
cache in a graph.

Workaround: a block of N tokens is split into two pieces A, B (≤ P tokens each). Every layer has two caches X and Y holding
the same prefix, and two cache slots s1 < s2 after the prefix:
  head graph l: writes A at s1 of X_l and B at s1 of Y_l (their attention outputs are not used)
  step graph l: B at s2 of X_l → attends prefix + A + B  = B's rows of the original attention
             A at s2 of Y_l → attends prefix + B + A  = A's rows
                then the rest of layer l.
Keys carry their RoPE, so the order of A and B in a cache does not change the result; the masks (one per cache) leave out
padding and other requests' positions. A paged op can be called once per cache in a graph, so every layer has a head graph and a step graph (64 calls per pass).
Hidden states between graphs are bf16 (as in the original bf16 model). Measured 10-14: writing layer l+1 from the step
graph (4 paged calls) does not compile at 2P = 16896; rbln_cache_update instead of the write calls put every cache on
chiplet 0 and gave a different output on every run.
"""

from __future__ import annotations

import torch
from torch import nn

from .dit_kv import KVConfig, rope_real


SPLIT_P = 8448  # piece size: a 2048-class block (≤ 16896 tokens) in two pieces
# One piece (+ a text chunk) per partition. The flash op's Init scratch grows with query chunk × visible cache and adds up
# over graphs (measured 10-14, against 35 GiB on chiplet 0 and ~64 GiB of Init + Intermediate per device), so the cache is
# kept at 6 partitions: e.g. a 2048² target + two output_resolution-2048 condition images, or + 6 references of 4352
# tokens. The write calls see only their own partition (block table of one).
SPLIT_CFG = KVConfig(part=SPLIT_P + 512, n_part=6, attn="flash")


def _paged(sh, cfg, q, k, v, kc, vc, pos, bt, mask, bidir=True):
    """Paged flash attention over the partitions listed in `bt` (positions `pos` are logical: index into bt × part)."""
    C = q.shape[1]
    H, D = cfg.heads, cfg.head_dim
    o = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
        q=q.transpose(1, 2).reshape(1, H, 1, C, D),
        k=k.transpose(1, 2).unsqueeze(2),
        v=v.transpose(1, 2).unsqueeze(2),
        kcache=kc.unsqueeze(2),
        vcache=vc.unsqueeze(2),
        seq=pos[:, 0].unsqueeze(-1),
        scale=sh.scale,
        block_table=bt,
        block_size=cfg.part,
        is_bidirectional=bidir,
        mask=mask,
        partition=cfg.part,
    )
    return o.view(1, H, C, D).transpose(1, 2)  # [1, C, H, D]


class _Base(nn.Module):
    def __init__(self, shared, cfg: KVConfig):
        super().__init__()
        self.sh, self.cfg = shared, cfg

    @property
    def m(self):
        return self.sh.m

    @staticmethod
    def _mod(mod):  # mod [1, 4D] from DiTSplitMod
        s1, g1, s2, g2 = mod[:, None].chunk(4, dim=-1)
        return s1, g1.tanh(), s2, g2.tanh()

    def _qkv(self, blk, x, s1, cos, sin):
        attn, H = blk.attn, self.cfg.heads
        h = blk.img_norm1(x) * (1 + s1)
        q = attn.norm_q(attn.to_q(h).unflatten(-1, (H, -1)))
        k = attn.norm_k(attn.to_k(h).unflatten(-1, (H, -1)))
        v = attn.to_v(h).unflatten(-1, (H, -1))
        return rope_real(q.to(v.dtype), cos, sin), rope_real(k.to(v.dtype), cos, sin), v


class DiTSplitMod(_Base):
    """Timestep → modulation [1, 4D] and output-norm scale [1, D]. The first graph of a compile context: the other graphs
    take these as inputs (when a later graph of the context used a shared weight differently from the first one, e.g.
    only part of the modulation, it failed with OpInvalidWeightSharingError, 10-14)."""

    def forward(self, t):
        m = self.m
        temb = m.time_text_embed.timestep_embedder(self.sh.tproj(t).to(t.dtype))
        return m.modulation(temb), m.norm_out.linear(m.norm_out.silu(temb))


class DiTSplitHead(_Base):
    """Writes layer l's K/V of a split block: A at slot 1 of X_l, B at slot 1 of Y_l (paged op; outputs not used).

    inputs: x [2P, 64 fp32 (l == 0: latents) | 4096 bf16], mod [1, 4D] (DiTSplitMod), rope_idx [3, 2P],
            pos0 [1, P] (slot 1 inside its partition), bt1 [1] int16 (that partition), own [1, part] (ones), X_l k/v, Y_l k/v
    outputs: hidden [2P, 4096] bf16 (layer 0: after img_in), check [2, 4096] (token means of the two write outputs)"""

    def __init__(self, shared, cfg, layer: int):
        super().__init__(shared, cfg)
        self.layer = layer
        self._blk = [shared.m.transformer_blocks[layer]]  # a list: parameters stay under `sh` (one weight group)

    @property
    def blk(self):
        return self._blk[0]

    def forward(self, x, mod, rope_idx, pos0, bt1, own, kx, vx, ky, vy):
        x = self.m.img_in(x) if self.layer == 0 else x.float()
        x = x[None]
        s1, *_ = self._mod(mod)
        cos, sin = self.sh.rope(rope_idx)
        q, k, v = self._qkv(self.blk, x, s1, cos, sin)
        P = q.shape[1] // 2
        # a block table of one partition (the slot's): the op then sees only that partition (its own chunk at offset
        # pos0 and whatever precedes it there; the output is not used), and its Init scratch is sized by one
        # partition instead of the whole cache
        wa = _paged(self.sh, self.cfg, q[:, :P], k[:, :P], v[:, :P], kx, vx, pos0, bt1, own)
        wb = _paged(self.sh, self.cfg, q[:, P:], k[:, P:], v[:, P:], ky, vy, pos0, bt1, own)
        out = x[0] if self.layer == 0 else x[0, :1]  # later layers: the input itself is the hidden state
        return out.to(torch.bfloat16), torch.cat([wa.mean(1).flatten(1), wb.mean(1).flatten(1)])


class DiTSplitStep(_Base):
    """Layer l of a split block (its K/V already at slot 1 of X_l / Y_l), plus the output projection on the last layer.

    inputs: h [2P, 4096] bf16, mod [1, 4D] (+ nscale [1, D] on the last layer), rope_idx [3, 2P], pos [1, P] (slot 2),
            block_tables, mask [2, CAP] (X, Y), X_l k/v, Y_l k/v
    outputs: hidden [2P, 4096] bf16 (noise_pred [2P, 64] fp32 after the last layer)"""

    def __init__(self, shared, cfg, layer: int, last: bool):
        super().__init__(shared, cfg)
        self.layer, self.last = layer, last
        self._blk = [shared.m.transformer_blocks[layer]]

    @property
    def blk(self):
        return self._blk[0]

    def forward(self, h, mod, *args):
        args = list(args)
        nscale = args.pop(0) if self.last else None
        rope_idx, pos, bt, mask, kx, vx, ky, vy = args
        x = h.float()[None]
        s1, g1, s2, g2 = self._mod(mod)
        cos, sin = self.sh.rope(rope_idx)
        q, k, v = self._qkv(self.blk, x, s1, cos, sin)
        P = q.shape[1] // 2
        ob = _paged(self.sh, self.cfg, q[:, P:], k[:, P:], v[:, P:], kx, vx, pos, bt, mask[0:1])  # B sees A + B
        oa = _paged(self.sh, self.cfg, q[:, :P], k[:, :P], v[:, :P], ky, vy, pos, bt, mask[1:2])  # A sees B + A
        o = torch.cat([oa, ob], dim=1)
        x = x + g1 * self.blk.attn.to_out[0](o.flatten(2, 3))
        x = x + g2 * self.blk.img_mlp(self.blk.img_norm2(x) * (1 + s2))
        if self.last:
            m = self.m
            return m.proj_out(m.norm_out.norm(x) * (1 + nscale[:, None]))[0]
        return x[0].to(torch.bfloat16)


class DiTDualPrefix(_Base):
    """Layers [l0, l1) for a prefix chunk that fits one piece (text: causal; image: bidirectional), written to both caches.

    inputs: x [C, 4096 (text) | 64 (image) at layer 0 | 4096], kind_tag [k] (k = 3 text, 2 image), mod [1, 4D], rope_idx [3, C],
            pos [1, C], block_tables, mask [2, CAP], (X k/v, Y k/v) per layer
    outputs: hidden [C, 4096] / [1, 64] dummy after the last layer"""

    def __init__(self, shared, cfg, l0: int, l1: int, n_layers: int):
        super().__init__(shared, cfg)
        self.l0, self.l1, self.n_layers = l0, l1, n_layers
        self.blocks = list(shared.m.transformer_blocks[l0:l1])

    def forward(self, x, kind_tag, mod, rope_idx, pos, bt, mask, *caches):
        txt = kind_tag.shape[0] == 3
        if self.l0 == 0:
            x = self.m.txt_in(x) if txt else self.m.img_in(x)
        x = x[None] + kind_tag[:1] * 0.0  # (kind_tag only selects the executor)
        s1, g1, s2, g2 = self._mod(mod)  # prefix: modulation of t = 0
        cos, sin = self.sh.rope(rope_idx)
        for i, blk in enumerate(self.blocks):
            kx, vx, ky, vy = caches[4 * i : 4 * i + 4]
            q, k, v = self._qkv(blk, x, s1, cos, sin)
            o = _paged(self.sh, self.cfg, q, k, v, kx, vx, pos, bt, mask[0:1], not txt)
            oy = _paged(self.sh, self.cfg, q, k, v, ky, vy, pos, bt, mask[1:2], not txt)
            o = (o + oy) * 0.5  # same keys and values in both caches (only the A/B order differs); keeps the Y write
            x = x + g1 * blk.attn.to_out[0](o.flatten(2, 3))
            x = x + g2 * blk.img_mlp(blk.img_norm2(x) * (1 + s2))
        if self.l1 == self.n_layers:
            return x[0, :1, :1].repeat(1, 64)
        return x[0]
