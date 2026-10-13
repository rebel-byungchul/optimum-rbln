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

"""Qwen-Image-2.1 DiT — device-resident KV cache + chunked prefix processing (RBLN).

The joint sequence of the original (diffusers `QwenImage21Transformer2DModel`) is
  [text | cond image 1 | text | ... | cond image n | text | target image]
and attention is block-causal (causal over text, bidirectional within an image block). Text and condition image tokens (prefix)
are modulated with t=0 and are therefore step-independent; the original also extracts the prefix K/V at the first step and only
recomputes the target in later steps (cached). Here the same structure is ported as-is to NPU graphs.

- One chunk = a run of tokens of the same kind (text / image block / target). Each kind has its own graph.
- Attention uses `rbln_custom_ops.paged_flash_causal_attn_prefill`: it writes its own k,v at cache positions seq and
  attends over [cache prefix + own chunk]. Text uses is_bidirectional=False (causal); images and the target use True.
  mask[1, CAP] restricts attention to valid cache positions.
- CR13 constraint (measured 2026-10-13): flash is correct only if a chunk lies within a single partition (17408-token
  partitions with a 16896-token chunk were also verified).
  A bidirectional chunk longer than a partition is split into two pieces A, B and computed as [write A → write B (B sees A and B) → write A again
  after B (the original A position is masked out)], which reproduces the original bidirectional attention.
- RoPE keeps the cos/sin of the original table (QwenImage21Rope.freqs) as constants and gathers them by integer index (inside the graph).
- Putting all 32 layers in one graph fails to compile for large sequences (DEVICE_GRAPH_CONVERSION), so layers are split into groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ops import ExactTimestepProj


ROPE_TABLE_ROWS = 8192 + 1024  # original QwenImage21Rope: positive 0..8191, negative -1024..-1


def rope_index_to_row(idx: torch.Tensor) -> torch.Tensor:
    """Map original indexing (negative indices count from the end) to a table row index."""
    return torch.where(idx < 0, idx + ROPE_TABLE_ROWS, idx)


@dataclass
class KVConfig:
    """attn="flash": paged_flash_causal_attn_prefill, block = partition, each chunk within one partition.
    attn="eager": paged_causal_attn_prefill, single block (part=CAP), no chunk length limit, CAP < 32768 (int16 positions).
    A paged op can be called only once per cache within a graph (calling it twice crashes the compiler, 10-13)."""

    part: int  # block size (tokens)
    n_part: int  # number of blocks
    attn: str = "flash"
    heads: int = 32
    head_dim: int = 128

    @property
    def cap(self) -> int:
        return self.part * self.n_part


class RopeGather(nn.Module):
    """Keep the original RoPE table as real cos/sin and gather with [3, C] indices → [C, 128] (pairwise-repeated layout)."""

    def __init__(self, pos_embed):
        super().__init__()
        cos, sin = [], []
        for f in pos_embed.freqs:  # [9216, d/2] complex64 per axis
            cos.append(nn.Embedding.from_pretrained(f.real.float().repeat_interleave(2, dim=-1), freeze=True))
            sin.append(nn.Embedding.from_pretrained(f.imag.float().repeat_interleave(2, dim=-1), freeze=True))
        self.cos = nn.ModuleList(cos)
        self.sin = nn.ModuleList(sin)

    def forward(self, idx):  # idx [3, C] int64 (table row indices)
        c = torch.cat([self.cos[a](idx[a]) for a in range(3)], dim=-1)
        s = torch.cat([self.sin[a](idx[a]) for a in range(3)], dim=-1)
        return c, s


def rope_real(x, cos, sin):
    """Same result as the original apply_rotary_emb_qwen(use_real=False). x [1, C, H, D], cos/sin [C, D]."""
    x_pair = x.float().unflatten(-1, (-1, 2))
    x_rot = torch.stack([-x_pair[..., 1], x_pair[..., 0]], dim=-1).flatten(-2)
    return (x.float() * cos[None, :, None, :] + x_rot * sin[None, :, None, :]).to(x.dtype)


class DiTKVShared(nn.Module):
    """Auxiliary tensors shared by the graphs (RoPE table, timestep table, attention scale). The compiler groups a module's parameters/buffers
    into a weight group and shares device weights within a group, so all graphs must reference this single object."""

    def __init__(self, model, head_dim: int = 128):
        super().__init__()
        self.m = model
        self.rope = RopeGather(model.pos_embed)
        self.tproj = ExactTimestepProj(timestep_dim=256)
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=False)


class DiTKVSegment(nn.Module):
    """One graph that runs DiT layers [l0, l1).

    kind: "txt" (text prefix, causal) | "img" (condition image prefix, bidirectional) | "tgt" (target, bidirectional, every step)
    first: whether the input embedding (txt_in / img_in) is included. last: (tgt only) includes norm_out/proj_out.
    split: number of pieces a bidirectional chunk is split into to fit a partition (1 or 2).

    inputs (in order)
      x            first: txt [C, 4096] | img·tgt [C, 64] / otherwise hidden [C, 4096]
      timestep     [1] float32 (tgt only; prefix is modulated with t=0 like the original)
      rope_idx     [3, C] int64
      cache_pos    split==1: [1, C] int32 / split==2: [3, C/2] int32 (A, B, A')
      block_tables [n_part] int16
      mask         split==1: [1, CAP] / split==2: [2, CAP] (1st: for the A·B calls, 2nd: for the A' call)
      kc_l, vc_l   static cache per layer [n_part, H, part, D]
    output: hidden [C, 4096] (if not last) / noise_pred [C, 64] (tgt·last) / small dummy (prefix·last)
    """

    def __init__(self, shared, l0: int, l1: int, kind: str, first: bool, last: bool, cfg: KVConfig, split: int = 1):
        super().__init__()
        assert kind in ("txt", "img", "tgt") and split in (1, 2)
        assert not (split == 2 and kind == "txt"), "text is causal; split it into multiple chunks instead"
        if not isinstance(shared, DiTKVShared):  # model passed directly (for testing)
            shared = DiTKVShared(shared, cfg.head_dim)
        self.sh = shared
        model = shared.m
        self.blocks = list(model.transformer_blocks[l0:l1])
        self.l0, self.l1 = l0, l1
        self.kind, self.first, self.last, self.split = kind, first, last, split
        self.cfg = cfg

    @property
    def m(self):
        return self.sh.m

    @property
    def rope(self):
        return self.sh.rope

    @property
    def tproj(self):
        return self.sh.tproj

    @property
    def scale(self):
        return self.sh.scale

    # ---- modulation: all tokens in a chunk use the same t (t=0 for prefix, the actual t for target)
    def _temb(self, timestep, like):
        if self.kind == "tgt" or getattr(self, "t_input", False):
            t = timestep.to(like.dtype)
        else:
            t = torch.zeros(1, dtype=like.dtype)
        return self.m.time_text_embed.timestep_embedder(self.tproj(t).to(like.dtype))  # [1, D]

    def _paged(self, q, k, v, kc, vc, pos, block_tables, mask, bidir):
        C = q.shape[1]
        H, D = self.cfg.heads, self.cfg.head_dim
        args = {
            "q": q.transpose(1, 2).reshape(1, H, 1, C, D),
            "k": k.transpose(1, 2).unsqueeze(2),
            "v": v.transpose(1, 2).unsqueeze(2),
            "kcache": kc.unsqueeze(2),
            "vcache": vc.unsqueeze(2),
            "scale": self.scale,
            "block_table": block_tables,
            "block_size": self.cfg.part,
            "is_bidirectional": bidir,
            "mask": mask,
        }
        if self.cfg.attn == "flash":
            o = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
                seq=pos[:, 0].unsqueeze(-1), partition=self.cfg.part, **args
            )
        else:
            o = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(seq=pos[:, :1], **args)
        return o.view(1, H, C, D).transpose(1, 2)  # [1, C, H, D]

    def _attn(self, attn, x, cos, sin, kc, vc, cache_pos, block_tables, mask):
        H = self.cfg.heads
        q = attn.norm_q(attn.to_q(x).unflatten(-1, (H, -1)))
        k = attn.norm_k(attn.to_k(x).unflatten(-1, (H, -1)))
        v = attn.to_v(x).unflatten(-1, (H, -1))
        q, k = rope_real(q.to(v.dtype), cos, sin), rope_real(k.to(v.dtype), cos, sin)  # [1, C, H, D]
        if self.split == 1:
            o = self._paged(q, k, v, kc, vc, cache_pos[:1], block_tables, mask[:1], self.kind != "txt")
        else:
            h = q.shape[1] // 2
            qa, qb, ka, kb, va, vb = q[:, :h], q[:, h:], k[:, :h], k[:, h:], v[:, :h], v[:, h:]
            self._paged(qa, ka, va, kc, vc, cache_pos[0:1], block_tables, mask[0:1], True)  # write A
            ob = self._paged(qb, kb, vb, kc, vc, cache_pos[1:2], block_tables, mask[0:1], True)  # B: sees A·B
            oa = self._paged(
                qa, ka, va, kc, vc, cache_pos[2:3], block_tables, mask[1:2], True
            )  # A': original A masked out
            o = torch.cat([oa, ob], dim=1)
        return attn.to_out[0](o.flatten(2, 3))

    def forward(self, x, *args):
        args = list(args)
        timestep = args.pop(0) if (self.kind == "tgt" or getattr(self, "t_input", False)) else None
        rope_idx, cache_pos, block_tables, mask = args[:4]
        caches = args[4:]
        m = self.m
        if self.first:
            x = m.txt_in(x) if self.kind == "txt" else m.img_in(x)
        x = x[None]  # [1, C, D]
        temb = self._temb(timestep, x)
        mod = m.modulation(temb)[:, None]  # [1, 1, 4D]
        s1, g1, s2, g2 = mod.chunk(4, dim=-1)
        g1, g2 = g1.tanh(), g2.tanh()
        cos, sin = self.rope(rope_idx)
        for i, blk in enumerate(self.blocks):
            kc, vc = caches[2 * i], caches[2 * i + 1]
            h = blk.img_norm1(x) * (1 + s1)
            x = x + g1 * self._attn(blk.attn, h, cos, sin, kc, vc, cache_pos, block_tables, mask)
            h = blk.img_norm2(x) * (1 + s2)
            x = x + g2 * blk.img_mlp(h)
        if not self.last:
            return x[0]
        if self.kind == "tgt":
            scale = m.norm_out.linear(m.norm_out.silu(temb).to(x.dtype))[:, None]
            x = m.norm_out.norm(x) * (1 + scale)
            return m.proj_out(x)[0]
        return x[0, :1, :8]  # last prefix piece: only the cache write matters, so return a small dummy output


# ---------------------------------------------------------------------------------------------------- host
# The code below only builds integer metadata (chunk boundaries, RoPE indices, cache positions, valid mask). No model computation.


@dataclass
class Segment:
    kind: str  # txt | img | tgt
    start: int  # start position in the joint sequence
    length: int
    rope_rows: torch.Tensor  # [3, length] int64


def build_segments(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]) -> list[Segment]:
    """Split the joint sequence into chunks and build RoPE indices following the same rules as the original QwenImage21Rope.forward.

    image_pad_mask: [S] bool (after expanding VLM slots ×4, target included). img_shapes: condition images + target (last).
    """
    is_img = image_pad_mask.tolist()
    total = len(is_img)
    segs, cursor, position = [], 0, 0
    for bi, (_, hh, ww) in enumerate(img_shapes):
        block_start = is_img.index(True, cursor)
        tl = block_start - cursor
        if tl > 0:
            fr = torch.arange(position, position + tl)
            segs.append(Segment("txt", cursor, tl, torch.stack([fr, fr, fr])))
        position += tl
        n = hh * ww
        fr = torch.full((n,), position)
        hi = torch.tensor([h for h in range(-(hh - hh // 2), hh // 2) for _ in range(ww)])
        wi = torch.tensor([w for _ in range(hh) for w in range(-(ww - ww // 2), ww // 2)])
        kind = "tgt" if bi == len(img_shapes) - 1 else "img"
        segs.append(Segment(kind, block_start, n, torch.stack([fr, hi, wi])))
        position += max(hh, ww)
        cursor = block_start + n
    if cursor < total:
        raise NotImplementedError(
            "batches with text after the target are not supported (this does not occur in the original pipeline either)"
        )
    for s in segs:
        s.rope_rows = rope_index_to_row(s.rope_rows.long())
    return segs


class CacheLayout:
    """Place chunks in the cache without crossing partition boundaries. Returns per chunk [(cache start, length), ...] (per piece when split)."""

    def __init__(self, cfg: KVConfig):
        self.cfg = cfg
        self.cursor = 0

    def place(self, length: int) -> int:
        part = self.cfg.part
        if self.cfg.attn == "eager":  # single block: no boundaries
            start = self.cursor
            self.cursor += length
            if self.cursor > self.cfg.cap:
                raise ValueError(f"KV cache capacity {self.cfg.cap} exceeded")
            return start
        if length > part:
            raise ValueError(f"chunk length {length} > partition {part}")
        p0 = self.cursor // part
        if (self.cursor + length - 1) // part != p0:  # if it crosses a boundary, start at the next partition
            self.cursor = (p0 + 1) * part
        start = self.cursor
        self.cursor += length
        if self.cursor > self.cfg.cap:
            raise ValueError(f"KV cache capacity {self.cfg.cap} exceeded")
        return start


KIND_TAG = {
    "tgt": 1,
    "img": 2,
    "txt": 3,
}  # the kind is encoded by the length of the kind_tag input (for target, its value is the timestep)


class DiTKVUnified(nn.Module):
    """DiT layer group that selects the kind (txt/img/tgt) by the length of the `kind_tag` input. Bundling multiple (kind, bucket) input
    shapes as executors of one compiled model shares weights and scratch across executors (loading the graphs separately allocates
    scratch per graph and runs out of device memory, 10-13).

    inputs: x [C, D_in] (first group: txt 4096 / img·tgt 64, otherwise hidden 4096), kind_tag [k] float32 ([timestep] for tgt),
            rope_idx [3, C] int64, cache_pos [1, C] int32, block_tables [n_part] int16, mask [1, CAP], kc, vc per layer
    output: hidden [C, 4096] if not the last group / last group: tgt → noise_pred [C, 64], prefix → [1, 64] dummy
    """

    def __init__(self, shared, l0, l1, n_layers, cfg: KVConfig):
        super().__init__()
        self.sh, self.l0, self.l1, self.cfg = shared, l0, l1, cfg
        self.first, self.last = l0 == 0, l1 == n_layers
        self.blocks = list(shared.m.transformer_blocks[l0:l1])

    def forward(self, x, kind_tag, rope_idx, cache_pos, block_tables, mask, *caches):
        kind = {1: "tgt", 2: "img", 3: "txt"}[kind_tag.shape[0]]
        seg = DiTKVSegment.__new__(DiTKVSegment)
        nn.Module.__init__(seg)
        seg.sh, seg.blocks, seg.l0, seg.l1 = self.sh, self.blocks, self.l0, self.l1
        seg.kind, seg.first, seg.split, seg.cfg = kind, self.first, 1, self.cfg
        seg.last = self.last
        # prefix must also use kind_tag so all executors have the same number of inputs (an unused input gets dropped). Multiplying by 0 gives t=0.
        t = kind_tag[:1] if kind == "tgt" else kind_tag[:1] * 0.0
        seg.t_input = True  # prefix is also modulated with the input t (=0)
        args = [t, rope_idx, cache_pos, block_tables, mask, *caches]
        out = DiTKVSegment.forward(seg, x, *args)
        if self.last and kind != "tgt":
            out = out[:, :1].reshape(1, -1)[:, :1].repeat(1, 64)  # unify the output shape to [1, 64]
        return out
