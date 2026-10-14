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

"""Compile/run the split DiT ("xxl" cache configuration, see dit_split.py): requests whose target or condition images go
beyond the flash chunk limit next to a long prefix (2048-class outputs with several references or with output_resolution
2048 condition images).

Bank layout: one compile context per layer half (the halves sit on different devices; a context's shared weights live in its
first file, so a half must not reference another half's files). Per half, in load order:
  mod               timestep → modulation (the other graphs take it as an input)
  head{l}           writes the K/V of layer l for a split block (layer 0: from the latents)
  step{l}           layer l of a split block (+ the output projection after the last layer)
  dual{txt,img}{a}-{b}  text / small image prefix chunks, written to both caches (loaded for the prefix phase only)
"""

from __future__ import annotations

import json
import os
import time

import torch

from .dit_kv import KIND_TAG, DiTKVShared, KVConfig
from .dit_split import SPLIT_CFG, SPLIT_P, DiTDualPrefix, DiTSplitHead, DiTSplitMod, DiTSplitStep


HIDDEN, LATENT_C = 4096, 64
# Prefix chunks that are not split, written to both caches by models loaded during the prefix phase only (next
# to the resident split models chiplet 0 has ~2 GiB left, measured 10-14): text (txt 64: the short text between
# references) and condition images up to 4352 tokens (a 1024-class reference; a split block would take two partitions).
DUAL_FAMILIES = {"txt": [("txt", 64), ("txt", 512)], "img": [("img", 4352)]}
DUAL_EXECUTORS = [x for v in DUAL_FAMILIES.values() for x in v]
DUAL_LAYERS = 4  # per dual model (both families of a range are loaded together: ~3 GiB of chiplet 0)
SHARE_WEIGHTS = False


def split_cache_names(layer):
    return [f"xxl_kx{layer}", f"xxl_vx{layer}", f"xxl_ky{layer}", f"xxl_vy{layer}"]


def compile_split(
    model,
    out_dir,
    l0,
    l1,
    n_layers,
    cfg: KVConfig = SPLIT_CFG,
    P: int = SPLIT_P,
    npu="RBLN-CR13",
    log=print,
    dual_layers=DUAL_LAYERS,
):
    """Layers [l0, l1) of the split DiT in one compile context (shared caches and weights); writes xxl_L{l0}-{l1}.json."""
    import rebel
    from rebel.compile_context import CompileContext

    os.makedirs(out_dir, exist_ok=True)
    name = f"xxl_L{l0}-{l1}"
    rec_path = os.path.join(out_dir, name + ".json")
    if os.path.exists(rec_path):
        return
    # caches are shared through the context; weights are not: with weight sharing, every graph binds the weights it reuses
    # from another file as inputs on chiplet 0 (~0.8 GiB per layer for head + step), without it they are constants
    # spread over the chiplets like the rest (measured 10-14)
    ctx = CompileContext(use_weight_sharing=SHARE_WEIGHTS)
    caches = {}
    for li in range(l0, l1):
        for nm in split_cache_names(li):
            t = torch.zeros(cfg.n_part, cfg.heads, cfg.part, cfg.head_dim)
            ctx.mark_static_address(t, nm)
            caches[nm] = t
    shared = DiTKVShared(model)
    common = lambda C, rows: (  # noqa: E731
        [
            ("rope_idx", [3, C], "int64"),
            ("pos", [rows, C // rows if rows == 2 else C], "int32"),
            ("block_tables", [cfg.n_part], "int16"),
            ("mask", [2, cfg.cap], "float32"),
        ],
        [
            torch.zeros(3, C, dtype=torch.long),
            torch.arange(C, dtype=torch.int32).view(rows, -1)
            if rows == 2
            else torch.arange(C, dtype=torch.int32)[None],
            torch.arange(cfg.n_part, dtype=torch.int16),
            torch.ones(2, cfg.cap),
        ],
    )

    def cache_io(layers):
        info, ex = [], []
        for li in layers:
            for nm in split_cache_names(li):
                info.append((nm, list(caches[nm].shape), "float32"))
                ex.append(caches[nm])
        return info, ex

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

    C2 = 2 * P
    D4 = 4 * HIDDEN
    comp(f"{name}.mod", DiTSplitMod(shared, cfg), [[("t", [1], "float32")]], [[torch.tensor([0.5])]], "mod")
    rope_i = [("rope_idx", [3, C2], "int64"), ("pos", [1, P], "int32")]
    rope_e = [torch.zeros(3, C2, dtype=torch.long), torch.arange(P, dtype=torch.int32)[None]]
    tail_i = [("block_tables", [cfg.n_part], "int16"), ("mask", [2, cfg.cap], "float32")]
    tail_e = [torch.arange(cfg.n_part, dtype=torch.int16), torch.ones(2, cfg.cap)]
    for li in range(l0, l1):
        last = li + 1 == n_layers
        xd, xt = (LATENT_C, "float32") if li == 0 else (HIDDEN, "bfloat16")
        i2, e2 = cache_io([li])
        info = [("h", [C2, HIDDEN], "bfloat16"), ("mod", [1, D4], "float32")]
        ex = [torch.zeros(C2, HIDDEN, dtype=torch.bfloat16), torch.zeros(1, D4)]
        if last:
            info.append(("nscale", [1, HIDDEN], "float32"))
            ex.append(torch.zeros(1, HIDDEN))
        comp(
            f"{name}.step{li}",
            DiTSplitStep(shared, cfg, li, last),
            [info + rope_i + tail_i + i2],
            [ex + rope_e + tail_e + e2],
            "step",
            layer=li,
        )
        # after the step graph: the step graph (full block table) has to be the first to declare the caches;
        # with the head graph first (one-partition block table) the files did not load (LOADING_FAILED, 10-14)
        comp(
            f"{name}.head{li}",
            DiTSplitHead(shared, cfg, li),
            [
                [
                    ("x", [C2, xd], xt),
                    ("mod", [1, D4], "float32"),
                    rope_i[0],
                    ("pos0", [1, P], "int32"),
                    ("bt1", [1], "int16"),
                    ("own", [1, cfg.part], "float32"),
                    *i2,
                ]
            ],
            [
                [
                    torch.zeros(C2, xd, dtype=getattr(torch, xt)),
                    torch.zeros(1, D4),
                    rope_e[0],
                    torch.arange(P, dtype=torch.int32)[None],
                    torch.zeros(1, dtype=torch.int16),
                    torch.ones(1, cfg.part),
                    *e2,
                ]
            ],
            "head",
            layer=li,
        )
    for a in range(l0, l1, dual_layers):
        b = min(a + dual_layers, l1)
        for fam, execs in DUAL_FAMILIES.items():
            infos, exs = [], []
            for kind, C in execs:
                ci1, ce1 = common(C, 1)
                xd = (HIDDEN if kind == "txt" else LATENT_C) if a == 0 else HIDDEN
                i2, e2 = cache_io(range(a, b))
                infos.append(
                    [
                        ("x", [C, xd], "float32"),
                        ("kind_tag", [KIND_TAG[kind]], "float32"),
                        ("mod", [1, D4], "float32"),
                        *ci1,
                        *i2,
                    ]
                )
                exs.append([torch.zeros(C, xd), torch.zeros(KIND_TAG[kind]), torch.zeros(1, D4), *ce1, *e2])
            comp(
                f"{name}.dual{fam}{a}-{b}",
                DiTDualPrefix(shared, cfg, a, b, n_layers),
                infos,
                exs,
                "dual",
                l0=a,
                l1=b,
                family=fam,
                executors=[list(x) for x in execs],
            )
    rec = {"name": name, "cfg": "xxl", "split": True, "l0": l0, "l1": l1, "P": P, "kv": vars(cfg), "files": files}
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2)
