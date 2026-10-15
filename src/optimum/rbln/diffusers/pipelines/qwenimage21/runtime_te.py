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

"""Compile/load the text encoder KV graph set, and the encoding runtime."""

from __future__ import annotations

import json
import os
import time

import torch

from .runtime_aux import run_verified
from .te_kv import N_DEEPSTACK, TEConfig, TEKVSegment, TEKVShared


NUM_LAYERS = 36
HIDDEN = 4096
# (kind, chunk bucket, layers per graph). The language model is causal, so splitting the prompt into chunks gives the same
# result; text-only chunks use 256 tokens (no 2048-token text graph: its 7.4 GiB of scratch would not fit next to the
# 2048-class DiT, see runtime_dit.UNIFIED), chunks that contain image tokens use the 2048-token multimodal graph.
GRAPHS = [("txt", 256, 36), ("mm", 2048, 36)]
CFG = TEConfig()


def _ranges(per):
    return [(l0, min(NUM_LAYERS, l0 + per)) for l0 in range(0, NUM_LAYERS, per)]


def compile_te_bank(lm, out_dir, graphs=GRAPHS, npu="RBLN-CR13", log=print):
    import rebel
    from rebel.compile_context import CompileContext

    os.makedirs(out_dir, exist_ok=True)
    ctx = CompileContext(use_weight_sharing=True)
    caches = {}
    for li in range(NUM_LAYERS):
        for nm in (f"te_k{li}", f"te_v{li}"):
            t = torch.zeros(1, CFG.kv_heads, CFG.cap, CFG.head_dim)
            ctx.mark_static_address(t, nm)
            caches[nm] = t
    mpath = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"graphs": []}
    done = {(g["kind"], g["C"]) for g in manifest["graphs"]}
    shared = TEKVShared(lm, CFG)
    for kind, C, per in graphs:
        if (kind, C) in done:
            continue
        while True:
            try:
                entries = []
                for l0, l1 in _ranges(per):
                    name = f"te_{kind}_{C}_L{l0}-{l1}"
                    seg = TEKVSegment(shared, l0, l1, kind, CFG).eval()
                    if l0 == 0:
                        info, ex = [("input_ids", [C], "int64")], [torch.zeros(C, dtype=torch.long)]
                        if kind == "mm":
                            info += [("vis", [C, HIDDEN], "float32"), ("is_vis", [C], "float32")]
                            ex += [torch.zeros(C, HIDDEN), torch.zeros(C)]
                    else:
                        info, ex = [("hidden", [C, HIDDEN], "float32")], [torch.zeros(C, HIDDEN)]
                    if kind == "mm" and l0 < seg.n_ds:
                        info.append(("ds", [seg.n_ds, C, HIDDEN], "float32"))
                        ex.append(torch.zeros(seg.n_ds, C, HIDDEN))
                    info += [
                        ("pos", [3, C], "int64"),
                        ("cache_pos", [1, C], "int32"),
                        ("block_tables", [1], "int16"),
                        ("mask", [1, CFG.cap], "float32"),
                    ]
                    ex += [
                        torch.zeros(3, C, dtype=torch.long),
                        torch.arange(C, dtype=torch.int32)[None],
                        torch.zeros(1, dtype=torch.int16),
                        torch.ones(1, CFG.cap),
                    ]
                    for li in range(l0, l1):
                        for nm in (f"te_k{li}", f"te_v{li}"):
                            info.append((nm, list(caches[nm].shape), "float32"))
                            ex.append(caches[nm])
                    t0 = time.perf_counter()
                    c = rebel.compile_from_torch(seg, input_info=info, example_inputs=ex, compile_context=ctx, npu=npu)
                    c.save(os.path.join(out_dir, name + ".rbln"))
                    log(f"compiled {name} in {time.perf_counter() - t0:.0f}s")
                    entries.append({"name": name, "kind": kind, "C": C, "l0": l0, "l1": l1})
                break
            except Exception as e:  # noqa: BLE001
                if per <= 6:
                    raise
                log(f"FAILED te {kind}/{C} with {per} layers/graph: {str(e)[:80]} -> retry {per // 2}")
                per //= 2
        manifest["graphs"].extend(entries)
        json.dump(manifest, open(mpath, "w"), indent=2)
    return manifest


class RBLNTextEncoderKV:
    """input_ids (+ image features) → last layer output (before the final norm) [L, 4096]."""

    def __init__(self, bank_dir, device=0):
        m = json.load(open(os.path.join(bank_dir, "manifest.json")))
        wanted = {(k, c) for k, c, _ in GRAPHS}
        self.graphs = [g for g in m["graphs"] if (g["kind"], g["C"]) in wanted]
        from .runtime_aux import LazyRuntimes

        small = min((g for g in self.graphs if g["kind"] == "txt"), key=lambda g: g["C"])
        self.rt = LazyRuntimes(
            {g["name"]: os.path.join(bank_dir, g["name"] + ".rbln") for g in self.graphs},
            device,
            pinned=[g["name"] for g in self.graphs if g["kind"] == "txt" and g["C"] == small["C"]],
        )
        self.timings = {}
        self._buckets = {k: sorted({g["C"] for g in self.graphs if g["kind"] == k}) for k in ("txt", "mm")}
        self._chains = {}  # (kind, C) → graph records in layer order
        for g in sorted(self.graphs, key=lambda g: g["l0"]):
            self._chains.setdefault((g["kind"], g["C"]), []).append(g)

    def free(self):
        """Unload the runtimes to give the device memory to the DiT; the next encode loads them again."""
        self.rt.free()

    def _chain(self, kind, C):
        return [(g, self.rt[g["name"]]) for g in self._chains[(kind, C)]]

    def encode(self, input_ids, position_ids, vis_rows=None, vis_mask=None, deepstack=None):
        """input_ids [L] int64, position_ids [3, L] int64 (mRoPE), vis_mask [L] bool (image token positions),
        vis_rows [n_vis, 4096] (in vis_mask order), deepstack [3, n_vis, 4096]."""
        t_start = time.perf_counter()
        L = input_ids.shape[0]
        if L > CFG.cap:
            raise ValueError(f"text encoder input of {L} tokens > cache {CFG.cap}")
        mask = torch.zeros(1, CFG.cap)
        out = torch.zeros(L, HIDDEN)
        vis_full = ds_full = None
        if vis_mask is not None and vis_mask.any():
            vis_full = torch.zeros(L, HIDDEN)
            vis_full[vis_mask] = vis_rows.float()
            ds_full = torch.zeros(deepstack.shape[0], L, HIDDEN)
            ds_full[:, vis_mask] = deepstack.float()
        o = 0
        big_txt, big_mm = max(self._buckets["txt"]), max(self._buckets["mm"], default=0)
        while o < L:
            # a chunk that reaches image tokens uses the multimodal graph, otherwise the text graph
            has_vis = vis_full is not None and bool(vis_mask[o : o + big_mm].any())
            kind = "mm" if has_vis else "txt"
            n = min(big_mm if has_vis else big_txt, L - o)
            C = next(c for c in self._buckets[kind] if c >= n)
            ids = torch.zeros(C, dtype=torch.long)
            ids[:n] = input_ids[o : o + n]
            pos = torch.zeros(3, C, dtype=torch.long)
            pos[:, :n] = position_ids[:, o : o + n]
            mask[0, o : o + n] = 1
            cp = (o + torch.arange(C, dtype=torch.int32))[None]
            bt = torch.zeros(1, dtype=torch.int16)
            h = None
            for g, rt in self._chain(kind, C):
                args = [ids if g["l0"] == 0 else h]
                if g["l0"] == 0 and kind == "mm":
                    v = torch.zeros(C, HIDDEN)
                    v[:n] = vis_full[o : o + n]
                    m = torch.zeros(C)
                    m[:n] = vis_mask[o : o + n].float()
                    args += [v, m]
                if kind == "mm" and g["l0"] < N_DEEPSTACK:
                    d = torch.zeros(ds_full.shape[0], C, HIDDEN)
                    d[:, :n] = ds_full[:, o : o + n]
                    args.append(d)
                h = run_verified(  # a repeat rewrites the same cache rows
                    lambda rt=rt, args=args, pos=pos, cp=cp, bt=bt, mask=mask: rt.run(*args, pos, cp, bt, mask)
                )
            out[o : o + n] = h[:n]
            o += n
        self.timings.setdefault("te", []).append(time.perf_counter() - t_start)
        return out
