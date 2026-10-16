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

"""Compile/run the "xxl" DiT (dit_paged.py): per-layer weight graphs + shared attention graphs over a block cache.

Bank layout: one compile context per layer half (the halves sit on different devices). Per half, in load order:
  mod                 timestep → modulation, output-norm scale
  attn_{txt64,txt512,img,piece}   attention over the half's block cache (no weights, shared by every layer)
  pre{l0}_{block,txt} q/k/v of the half's first layer (a block of rows; layer 0 also 512 text rows through txt_in)
  mid{l}              rest of layer l + q/k/v of layer l+1 (the half's last layer: hidden state only)
  post{31}            last layer of the target + output projection (second half only)
"""

from __future__ import annotations

import json
import os
import time

import torch

from .dit_kv import DiTKVShared
from .dit_paged import (
    BLOCKS_PER_LAYER,
    CHUNKS,
    PART,
    PIECE,
    WEIGHT_C,
    DiTPagedMid,
    DiTPagedMod,
    DiTPagedPost,
    DiTPagedPre,
    PagedAttention,
)
from .ops import split_timestep


HIDDEN, LATENT_C, H, D = 4096, 64, 32, 128
IO_DTYPE = torch.float32  # tensors passed between the xxl graphs through the host (bf16 tripled the DiT error)


def compile_paged(
    model, out_dir, l0, l1, n_layers, npu="RBLN-CR13", log=print, blocks_per_layer=BLOCKS_PER_LAYER, io_dtype=IO_DTYPE
):
    """Layers [l0, l1) in one compile context (the block cache is shared by its graphs); writes xxl_L{l0}-{l1}.json.
    io_dtype: dtype of the tensors passed between the graphs (hidden state, q/k/v, attention output)."""
    import rebel
    from rebel.compile_context import CompileContext

    os.makedirs(out_dir, exist_ok=True)
    io = str(io_dtype).split(".")[-1]
    name = f"xxl_L{l0}-{l1}"
    rec_path = os.path.join(out_dir, name + ".json")
    if os.path.exists(rec_path):
        return
    nb = (l1 - l0) * blocks_per_layer
    nv = blocks_per_layer
    # no weight sharing (with it, every graph's weights land on chiplet 0); the layer graphs have one executor (all chunks
    # padded to a block) since without sharing each executor holds its own copy of the weights
    ctx = CompileContext(use_weight_sharing=False)
    kc = torch.zeros(nb, H, 1, PART, D)
    vc = torch.zeros(nb, H, 1, PART, D)
    ctx.mark_static_address(kc, "xxl_k")
    ctx.mark_static_address(vc, "xxl_v")
    shared = DiTKVShared(model)
    files = []

    def comp(fname, mod, infos, exs, role, **meta):
        path = os.path.join(out_dir, fname + ".rbln")
        t0 = time.perf_counter()
        if len(infos) == 1:
            infos, exs = infos[0], exs[0]
        c = rebel.compile_from_torch(mod.eval(), input_info=infos, example_inputs=exs, compile_context=ctx, npu=npu)
        c.save(path)
        c_by = c.get_alloc_per_chiplet_by_key()
        e = {
            "name": fname,
            "role": role,
            **meta,
            "compile_s": round(time.perf_counter() - t0, 1),
            "chiplet_gib": [round(sum(v[0][ch] for v in c_by.values()) / 2**30, 2) for ch in range(4)],
        }
        log(f"compiled {fname} in {e['compile_s']:.0f}s chiplets={e['chiplet_gib']}")
        files.append(e)

    D4 = 4 * HIDDEN
    comp(
        f"{name}.mod",
        DiTPagedMod(shared),
        [[("t", [3], "float32")]],
        [[split_timestep(torch.tensor([0.5]))[0]]],
        "mod",
    )
    for key, (C, bidir) in CHUNKS.items():
        info = [(n, [C, H, D], io) for n in ("q", "k", "v")]
        info += [("seq", [1, 1], "int32"), ("bt", [nv], "int16"), ("mask", [1, nv * PART], "float32")]
        info += [("xxl_k", list(kc.shape), "float32"), ("xxl_v", list(vc.shape), "float32")]
        ex = [torch.zeros(C, H, D, dtype=io_dtype) + i for i in range(3)]
        ex += [
            torch.zeros(1, 1, dtype=torch.int32),
            torch.arange(nv, dtype=torch.int16),
            torch.ones(1, nv * PART),
            kc,
            vc,
        ]
        comp(
            f"{name}.attn_{key}",
            PagedAttention(C, bidir, PART, io_dtype=io_dtype),
            [info],
            [ex],
            "attn",
            chunk=key,
            C=C,
        )

    def x_spec(key, C, layer):
        if layer == 0:
            xd = HIDDEN if key == "txt" else LATENT_C
            return ("x", [C, xd], "float32"), torch.zeros(C, xd)
        return ("x", [C, HIDDEN], io), torch.zeros(C, HIDDEN, dtype=io_dtype)

    def rope(C):
        return ("rope_idx", [3, C], "int64"), torch.zeros(3, C, dtype=torch.long)

    # first layer of the half: q/k/v from the input (layer 0: text rows through txt_in in 512-row chunks, latents through
    # img_in; the next half: the hidden state)
    CB = WEIGHT_C["block"]
    pre = [("block", CB)] + ([("txt", WEIGHT_C["txt"])] if l0 == 0 else [])
    for key, C in pre:
        (xi, xe), (ri, re) = x_spec(key, C, l0), rope(C)
        comp(
            f"{name}.pre{l0}_{key}",
            DiTPagedPre(shared, l0, key == "txt", io_dtype=io_dtype),
            [[xi, ("mod", [1, D4], "float32"), ri]],
            [[xe, torch.zeros(1, D4), re]],
            "pre",
            layer=l0,
            key=key,
        )
    for li in range(l0, l1):
        if li + 1 == n_layers:  # last layer: only the target needs it (prefix chunks end with their K/V writes)
            C = WEIGHT_C["block"]
            info = [
                ("h", [C, HIDDEN], io),
                ("o", [C, H, D], io),
                ("mod", [1, D4], "float32"),
                ("nscale", [1, HIDDEN], "float32"),
            ]
            ex = [
                torch.zeros(C, HIDDEN, dtype=io_dtype),
                torch.zeros(C, H, D, dtype=io_dtype),
                torch.zeros(1, D4),
                torch.zeros(1, HIDDEN),
            ]
            comp(f"{name}.post{li}", DiTPagedPost(shared, li, io_dtype=io_dtype), [info], [ex], "post", layer=li)
            continue
        ri, re = rope(CB)
        info = [("h", [CB, HIDDEN], io), ("o", [CB, H, D], io), ("mod", [1, D4], "float32"), ri]
        ex = [
            torch.zeros(CB, HIDDEN, dtype=io_dtype),
            torch.zeros(CB, H, D, dtype=io_dtype),
            torch.zeros(1, D4),
            re,
        ]
        comp(f"{name}.mid{li}", DiTPagedMid(shared, li, li + 1 < l1, io_dtype=io_dtype), [info], [ex], "mid", layer=li)
    rec = {
        "name": name,
        "cfg": "xxl",
        "paged": True,
        "l0": l0,
        "l1": l1,
        "n_layers": n_layers,
        "blocks_per_layer": blocks_per_layer,
        "part": PART,
        "piece": PIECE,
        "io_dtype": io,
        "files": files,
    }
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2)


# ---------------------------------------------------------------------------------------------------------------- host
class BlockLayout:
    """Token positions in a context's block region: chunks never cross a block; pieces fill most of a block."""

    def __init__(self, start_block, end_block, part=PART):
        self.part, self.start, self.end = part, start_block, end_block
        self.cursor = start_block * part

    def place(self, n):
        if self.cursor % self.part + n > self.part:
            self.cursor = (self.cursor // self.part + 1) * self.part
        s = self.cursor
        self.cursor += n
        if self.cursor > self.end * self.part:
            raise ValueError(f"xxl block cache exhausted ({self.end - self.start} blocks of {self.part} tokens)")
        return s

    def blocks_upto(self, pos):
        """Blocks of the region from its start to the block holding `pos`, in order."""
        return list(range(self.start, pos // self.part + 1))


class PagedDiT:
    """Runs requests on an xxl bank (see module docstring). Owned by RBLNDiTKV, which picks it when no other cache
    configuration holds the request; `run_group(rt, args, verify)` is RBLNDiTKV._run_group."""

    def __init__(self, halves, devices, run_group):
        self.halves = sorted(halves, key=lambda h: h[0]["l0"])  # [(record, dir)]
        self.devices = devices
        self.run_group = run_group
        self.verify_attention = False
        self.nbl = self.halves[0][0]["blocks_per_layer"]
        self.part = self.halves[0][0]["part"]
        self.P = self.halves[0][0]["piece"]
        self.n_layers = self.halves[0][0]["n_layers"]
        self.io = getattr(torch, self.halves[0][0].get("io_dtype", "bfloat16"))  # dtype passed between the graphs
        self.rts = None
        self._mods = {}  # float(t) → (modulation, output-norm scale): the same for every layer and request with that t

    # ---------------------------------------------------------------- loading
    def placement(self):
        out = []
        for i, (rec, d) in enumerate(self.halves):
            for fe in rec["files"]:
                out.append((os.path.join(d, fe["name"] + ".rbln"), self.devices[i % len(self.devices)], fe))
        return out

    def bind(self, rts):
        """rts: runtimes in placement() order."""
        self.rts = []
        it = iter(rts)
        for rec, _ in self.halves:
            g = {"attn": {}, "pre": {}, "mid": {}, "post": {}, "l0": rec["l0"], "l1": rec["l1"]}
            for fe in rec["files"]:
                rt = next(it)
                if fe["role"] == "mod":
                    g["mod"] = rt
                elif fe["role"] == "attn":
                    g["attn"][fe["chunk"]] = rt
                elif fe["role"] == "pre":
                    g["pre"][fe["key"]] = rt
                else:
                    g[fe["role"]][fe["layer"]] = rt
            self.rts.append(g)

    # ---------------------------------------------------------------- layout
    def capacity_check(self, segs, n_tgt):
        """Blocks a context's prefix needs (ValueError when the target or the prefix does not fit)."""
        if n_tgt > 2 * self.P:
            raise ValueError(f"target of {n_tgt} tokens > {2 * self.P}")
        lay = BlockLayout(0, self.nbl - 2, self.part)
        for kind, C, _, _ in self.chunk_plan(segs):
            lay.place(C if kind != "block" else self.P)
            if kind == "block":
                lay.place(self.P)
        return (lay.cursor + self.part - 1) // self.part

    def chunk_plan(self, segs):
        """[(kind, alloc, seg, row offset)] for the prefix: text in 512/64 chunks, images ≤ 4352 as one chunk, larger
        images as a split block."""
        out = []
        for s in segs:
            if s.kind == "tgt":
                continue
            if s.kind == "txt":
                T, t64 = CHUNKS["txt512"][0], CHUNKS["txt64"][0]
                for o in range(0, s.length, T):
                    n = min(T, s.length - o)
                    out.append(("txt64" if n <= t64 else "txt512", t64 if n <= t64 else T, s, o))
            elif s.length <= CHUNKS["img"][0]:
                out.append(("img", CHUNKS["img"][0], s, 0))
            elif s.length <= 2 * self.P:
                out.append(("block", 2 * self.P, s, 0))
            else:
                raise ValueError(f"condition image of {s.length} tokens > {2 * self.P}")
        return out

    # ---------------------------------------------------------------- calls
    def _attn_args(self, g, layer, blocks, seq, valid):
        """Block table (physical blocks of `layer`) and mask for a call that sees `blocks` in this order."""
        base = (layer - g["l0"]) * self.nbl
        bt = torch.full((self.nbl,), base, dtype=torch.int16)
        bt[: len(blocks)] = torch.tensor([base + b for b in blocks], dtype=torch.int16)
        key = (id(valid), tuple(blocks))  # the mask does not depend on the layer: built once per group of calls
        mask = self._masks.get(key)
        if mask is None:
            mask = torch.zeros(1, self.nbl * self.part)
            mask[0, : len(blocks) * self.part] = valid[list(blocks)].flatten()
            self._masks[key] = mask
        return torch.tensor([[seq]], dtype=torch.int32), bt, mask

    def _attend(self, g, layer, key, qkv, calls, valid):
        """calls: [(blocks, seq, rows)] — one attention call each on rows of q/k/v; returns the outputs per call."""
        outs = []
        for blocks, seq, rows in calls:
            q, k, v = (t[rows] for t in qkv)
            s, bt, mask = self._attn_args(g, layer, blocks, seq, valid)
            # attention graphs are a single op: run once (they never disagreed with themselves; the disagreements come
            # from large multi-op graphs, which stay verified)
            outs.append(self.run_group(g["attn"][key], (q, k, v, s, bt, mask), self.verify_attention))
        return outs

    def _layer_calls(self, kind, place):
        """Attention calls of a chunk at one layer: (attention graph key, [(blocks, seq, rows)], assemble(outputs)).
        place: {"blocks", "s"} for a single chunk at region position s (blocks: the region's blocks up to s's block),
        {"before", "sa", "sb"} for a split block (before: the blocks that precede piece A in logical order)."""
        part = self.part
        if kind != "block":
            blocks, s = place["blocks"], place["s"]
            return kind, [(blocks, (len(blocks) - 1) * part + s % part, slice(None))], lambda outs: outs[0]
        P, sa, sb, before = self.P, place["sa"], place["sb"], place["before"]
        ba, bb = sa // part, sb // part
        n = len(before)
        calls = [
            ([ba], sa % part, slice(0, P)),  # write A: its own block only (output not used; cost grows with the view)
            (before + [ba, bb], (n + 1) * part + sb % part, slice(P, 2 * P)),  # B attends prefix + A + B
            (before + [bb, ba], (n + 1) * part + sa % part, slice(0, P)),  # A attends prefix + B + A
        ]
        return "piece", calls, lambda outs: torch.cat([outs[2], outs[1]])

    def run_chunk(self, kind, x, rope, t, place, valid, target=False):
        """One chunk through every layer (see run_group); returns the noise prediction for the target."""
        return self.run_group_of([(kind, x, rope, place)], t, valid, target)

    def pack(self, chunks):
        """Consecutive prefix chunks → groups that share the layer graphs: small chunks are packed into one block of rows
        (the layer graphs work token by token), a split block is a group of its own."""
        groups, cur, used = [], [], 0
        for c in chunks:
            n = 2 * self.P if c[0] == "block" else CHUNKS[c[0]][0]
            if c[0] == "block" or used + n > 2 * self.P:
                if cur:
                    groups.append(cur)
                cur, used = [], 0
            if c[0] == "block":
                groups.append([c])
                continue
            cur.append(c)
            used += n
        if cur:
            groups.append(cur)
        return groups

    def run_group_of(self, chunks, t, valid, target=False):
        """Chunks [(kind, x, rope, place)] through every layer together: at each layer their attention calls run in order
        (a chunk sees the chunks before it at that layer, as when they run one by one), then one layer graph call
        computes all their rows. valid [blocks, part]: cache positions they may see. Returns the target's noise
        prediction (target=True: a single split block)."""
        CB = 2 * self.P
        self._masks = {}  # (valid, block order) → attention mask, for this group's calls (`valid` stays the same)
        spans, o0 = [], 0  # rows of each chunk in the packed block
        for kind, *_ in chunks:
            n = CB if kind == "block" else CHUNKS[kind][0]
            spans.append((o0, o0 + n))
            o0 += n
        rope = torch.zeros(3, CB, dtype=torch.long)
        for (a, _), (_, _, r, _) in zip(spans, chunks, strict=True):
            rope[:, a : a + r.shape[1]] = r
        plans = [self._layer_calls(kind, place) for kind, _, _, place in chunks]
        out = None
        H_ = None
        for g in self.rts:
            mod, nscale = self._mod(t)
            if g["l0"] == 0:  # embeddings: text rows through txt_in (one text chunk per call), the rest through img_in
                H_ = torch.zeros(CB, HIDDEN, dtype=self.io)
                qkv = [torch.zeros(CB, H, D, dtype=self.io) for _ in range(3)]
                img = torch.zeros(CB, LATENT_C)
                any_img = False
                for (a, b), (kind, x, r, _) in zip(spans, chunks, strict=True):
                    if kind.startswith("txt"):
                        T = WEIGHT_C["txt"]
                        xi = torch.zeros(T, x.shape[1])
                        xi[: x.shape[0]] = x
                        ri = torch.zeros(3, T, dtype=torch.long)
                        ri[:, : r.shape[1]] = r
                        res = self.run_group(g["pre"]["txt"], (xi, mod, ri), True, multi=True)
                        H_[a:b] = res[0][: b - a]
                        for dst, src in zip(qkv, res[1:], strict=True):
                            dst[a:b] = src[: b - a]
                    else:
                        img[a : a + x.shape[0]] = x
                        any_img = True
                if any_img:
                    res = self.run_group(g["pre"]["block"], (img, mod, rope), True, multi=True)
                    for (a, b), (kind, *_) in zip(spans, chunks, strict=True):
                        if not kind.startswith("txt"):
                            H_[a:b] = res[0][a:b]
                            for dst, src in zip(qkv, res[1:], strict=True):
                                dst[a:b] = src[a:b]
            else:
                H_, *qkv = self.run_group(g["pre"]["block"], (H_, mod, rope), True, multi=True)
            for li in range(g["l0"], g["l1"]):
                if len(chunks) == 1 and spans[0] == (0, CB):  # one block: its attention output is the whole row block
                    key, calls, assemble = plans[0]
                    o = assemble(self._attend(g, li, key, qkv, calls, valid))
                else:
                    o = torch.zeros(CB, H, D, dtype=self.io)
                    for (a, b), (key, calls, assemble) in zip(spans, plans, strict=True):
                        part_qkv = [t_[a:b] for t_ in qkv]
                        o[a:b] = assemble(self._attend(g, li, key, part_qkv, calls, valid))
                if li + 1 == self.n_layers:
                    if target:
                        out = self.run_group(g["post"][li], (H_, o, mod, nscale), True)
                    break
                res = self.run_group(g["mid"][li], (H_, o, mod, rope), True, multi=True)
                H_, qkv = res[0], res[1:]
        return out

    def _mod(self, t):
        """Modulation inputs of the layer graphs for timestep t (one small graph call per distinct t)."""
        key = float(t)
        if key not in self._mods:
            if len(self._mods) > 64:
                self._mods.clear()
            o = self.rts[0]["mod"].run(split_timestep(t.reshape(1))[0])
            self._mods[key] = (o[0], o[1])
        return self._mods[key]
