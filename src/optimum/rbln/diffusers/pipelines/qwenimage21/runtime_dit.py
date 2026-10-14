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

"""Compile/load the DiT-KV graph bank, and the runtime wrapper that replaces the transformer in the diffusers pipeline.

Two cache configurations, each compiled as 16-layer "unified" models (one model, several (kind, bucket) executors):
  eager: single block 32256 — default. text 512/4096, image and target buckets 4352/8448/16896 (up to the 2048 class).
         prefix + target ≤ 32256 (e.g. a 2048-class target with up to 3 condition images at the default output_resolution).
  flash: partition 16384 × 5, buckets 4352 — 1024-class targets with long prefixes (up to 10 reference images).
A request picks eager when it fits, else flash (`RBLNDiTKV._cfg_for`). Init/Intermediate scratch is allocated per executor
and per layer, so the eager halves (77 + 89 GiB) go to two devices; switching cfg unloads the other one on OOM.
(GRAPHS / compile_bank below are the older one-graph-per-kind bank, kept for the probes.)

Mapping to the original pipeline:
  kv_cache_mode="extract" (first step): write the prefix chunks to the cache and compute the target
  kv_cache_mode="cached"  (later steps): compute only the target (reuse prefix K/V)
  kv_cache_mode=None      (use_kv_cache=False): recompute from the prefix every step (same result)
  cache_context("cond"/"uncond"): the two CFG prefixes go in different cache regions. The target region is shared (overwritten on every call).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .dit_kv import KIND_TAG, CacheLayout, DiTKVSegment, DiTKVShared, DiTKVUnified, KVConfig, Segment, build_segments


CONFIGS = {
    "flash": KVConfig(part=16384, n_part=5, attn="flash"),
    "eager": KVConfig(part=32256, n_part=1, attn="eager"),
    # long prefixes: the last partition holds the target. Measured 10-14: attention across partitions is exact for chunks
    # up to 16320 tokens (also with 17408-token partitions) and wrong from 16384 tokens on (cos 0.86, whatever the
    # partition size), so the target and image buckets stop at 16320 (e.g. 2528x1648 = 16274 tokens).
    "xl": KVConfig(part=16384, n_part=3, attn="flash"),
}
# Unified model (per-kind/bucket executors in one compiled model): cfg → (executor list [(kind, C)], layer groups [(l0, l1)])
# flash serves the 1024 class (targets ≤ 4352 tokens, up to 10 references); eager serves larger targets (up to the 2048
# class, prefix + target < 32768). Only one of them is resident; a request that needs the other one swaps them.
# Device memory budget (measured 10-13/14, CR13): DRAM is split into 4 chiplets of 35 GiB and chiplet 0 receives most of the
# Init/Intermediate scratch (eager: 22.8 GiB of Intermediate on chiplet 0 vs 7.6 on the others), so chiplet 0 is the binding
# budget (`chiplet_gib` in the compile records). Scratch adds up over executors and layers, and storing in bf16 does not
# shrink it (activations are already 2-byte). Hence:
#   - text is causal, so long text is written in 512-token chunks (same result as one chunk);
#   - condition images are resized to output_resolution² (default 1024² → ≤ 4352 tokens), so one image bucket;
#   - eager has one target bucket (16896, the 2048 class); targets between 4352 and 16896 tokens are padded to it;
#   - eager is split in 8-layer models that RBLNDiTKV spreads over the devices by chiplet-0 budget.
UNIFIED = {
    "eager": (
        [("txt", 512), ("img", 4352), ("tgt", 16896)],
        [(0, 8), (8, 16), (16, 24), (24, 32)],
    ),
    "flash": ([("txt", 512), ("img", 4352), ("tgt", 4352)], [(0, 16), (16, 32)]),
    # xl keeps only the target executor resident; prefix chunks run through PREFIX_MODELS loaded for the prefix phase only
    "xl": ([("tgt", 16320)], [(0, 8), (8, 16), (16, 24), (24, 32)]),
}
# cfg → [(executors, layers per model)]: prefix-only models compiled in the same context as the resident model of each
# layer group (they share its caches and weights) and loaded only while a request writes its prefix. "big" ones also take
# condition images above 4352 tokens (output_resolution above the 1024 class).
PREFIX_MODELS = {
    # 8-layer prefix models: a load reads the whole file (mostly Init constants, ~4.6 GB each) from storage
    "xl": [([("txt", 512), ("img", 4352)], 8)],
    # eager keeps its prefix executors resident; condition images above 4352 tokens go through these
    # (buckets up to 14336: with the 16896 target region and some text that is all an eager cache of 32256 holds)
    "eager": [([("txt", 512), ("img", 4352), ("img", 8448), ("img", 12288), ("img", 14336)], 1)],
}
# (cfg, kind, token bucket, layers per graph)
GRAPHS = [
    ("flash", "txt", 512, 32),
    ("flash", "txt", 4096, 32),
    ("flash", "img", 4352, 32),
    ("flash", "img", 8448, 32),
    ("flash", "img", 16384, 16),
    ("flash", "tgt", 4352, 32),
    ("flash", "tgt", 8448, 32),
    ("flash", "tgt", 16384, 16),
    ("eager", "txt", 512, 32),
    ("eager", "txt", 4096, 32),
    ("eager", "img", 4352, 32),
    ("eager", "tgt", 16896, 16),
]
NUM_LAYERS = 32
HIDDEN = 4096
LATENT_C = 64


def graph_name(cfg, kind, C, l0, l1):
    return f"{cfg}_{kind}_{C}_L{l0}-{l1}"


def cache_names(cfg, layer):
    return f"{cfg}_k{layer}", f"{cfg}_v{layer}"


def _seg_ranges(per, n=None):
    n = n or NUM_LAYERS
    return [(l0, min(n, l0 + per)) for l0 in range(0, n, per)]


def compile_bank(model, out_dir: str, graphs=GRAPHS, npu="RBLN-CR13", log=print, num_layers=None):
    """Compile all graphs in one CompileContext and save them to out_dir. model is fp32 (= NPU dlfloat)."""
    from rebel.compile_context import CompileContext

    os.makedirs(out_dir, exist_ok=True)
    nl = num_layers or NUM_LAYERS
    ctx = CompileContext(use_weight_sharing=True)
    caches = {}
    for cname, cfg in CONFIGS.items():
        for li in range(nl):
            for nm in cache_names(cname, li):
                t = torch.zeros(cfg.n_part, cfg.heads, cfg.part, cfg.head_dim)
                ctx.mark_static_address(t, nm)
                caches[nm] = t
    manifest = {"configs": {k: vars(v) for k, v in CONFIGS.items()}, "graphs": [], "num_layers": nl}
    shared = DiTKVShared(model)  # all graphs in the same weight group → one copy of the weights on the device
    done = set()
    if os.path.exists(os.path.join(out_dir, "manifest.json")):  # resume compilation
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        done = {(g["cfg"], g["kind"], g["C"]) for g in manifest["graphs"]}
    for cname, kind, C, per in graphs:
        if (cname, kind, C) in done:
            continue
        cfg = CONFIGS[cname]
        while True:  # retry with fewer layers per graph
            try:
                entries = _compile_one(shared, ctx, caches, cfg, cname, kind, C, min(per, nl), out_dir, npu, log, nl)
                break
            except Exception as e:  # noqa: BLE001
                if per <= 4:
                    raise
                log(f"FAILED {cname}/{kind}/{C} with {per} layers/graph: {str(e)[:80]} -> retry {per // 2}")
                per //= 2
        manifest["graphs"].extend(entries)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def _compile_one(shared, ctx, caches, cfg, cname, kind, C, per, out_dir, npu, log, nl=NUM_LAYERS):
    import rebel

    entries = []
    if True:
        for l0, l1 in _seg_ranges(per, nl):
            name = graph_name(cname, kind, C, l0, l1)
            path = os.path.join(out_dir, name + ".rbln")
            seg = DiTKVSegment(shared, l0, l1, kind, l0 == 0, l1 == nl, cfg).eval()
            x_dim = (HIDDEN if kind == "txt" else LATENT_C) if l0 == 0 else HIDDEN
            info, ex = [("x", [C, x_dim], "float32")], [torch.zeros(C, x_dim)]
            if kind == "tgt":
                info.append(("timestep", [1], "float32"))
                ex.append(torch.tensor([0.5]))
            info += [
                ("rope_idx", [3, C], "int64"),
                ("cache_pos", [1, C], "int32"),
                ("block_tables", [cfg.n_part], "int16"),
                ("mask", [1, cfg.cap], "float32"),
            ]
            ex += [
                torch.zeros(3, C, dtype=torch.long),
                torch.arange(C, dtype=torch.int32)[None],
                torch.arange(cfg.n_part, dtype=torch.int16),
                torch.ones(1, cfg.cap),
            ]
            for li in range(l0, l1):
                for nm in cache_names(cname, li):
                    info.append((nm, list(caches[nm].shape), "float32"))
                    ex.append(caches[nm])
            t0 = time.perf_counter()
            c = rebel.compile_from_torch(seg, input_info=info, example_inputs=ex, compile_context=ctx, npu=npu)
            c.save(path)
            dt = time.perf_counter() - t0
            log(f"compiled {name} in {dt:.0f}s")
            entries.append(
                {"name": name, "cfg": cname, "kind": kind, "C": C, "l0": l0, "l1": l1, "compile_s": round(dt, 1)}
            )
    return entries


@dataclass
class _Placed:
    seg: Segment
    chunks: list = field(default_factory=list)  # [(graph C, joint start, length, cache start)]


class RBLNDiTKV:
    """Runtime that replaces `QwenImage21Transformer2DModel`. Mimics only the attributes the pipeline uses.

    bank_dir contains `{cfg}_L{l0}-{l1}.rbln` (+ .json) files built by compile_unified. One cfg = several layer-group models,
    one model = multiple (kind, bucket) executors (selected automatically by input shape).

    `device` may be a list: the layer-group models of a cfg are placed round-robin on those devices. Scratch memory (Init,
    Intermediate) is allocated per executor and per layer and is never shared, so the 2048-class eager models (77 + 89 GiB)
    only fit when the two 16-layer halves sit on different devices."""

    # flash first: it covers the 1024 class (targets ≤ 4352 tokens, up to 10 references) and leaves room for the VAE. eager
    # is loaded only for larger targets. One cfg is resident at a time: a switch unloads the other one (and cached VAE graphs)
    # before loading, since a load that fails half way does not give its device memory back (10-14).
    # xxl (split blocks, two caches per layer, one graph per layer) is the slowest and serves what the others cannot hold.
    PREFERENCE = ("flash", "eager", "xl", "xxl")

    def __init__(
        self, bank_dir, config: dict, device: int = 0, reserve: dict | None = None, local_cache: str | None = None
    ):
        dirs = [bank_dir] if isinstance(bank_dir, str) else list(bank_dir)
        self.bank_dir = dirs[0]
        self.models = {}  # cfg -> [(l0, l1, path, executors)]
        self.split_halves = []  # xxl: [(l0, l1, [(file record, path)])] (runtime_split.py)
        self.prefix_models = {}  # cfg -> [(family, l0, l1, path, executors)] (see PREFIX_MODELS)
        self.local_cache = local_cache  # local directory for copies of the prefix models (see _local)
        self.sizes = {}  # path -> chiplet-0 memory (GiB, from the compile record)
        self.reserve = dict(
            reserve or {}
        )  # device -> chiplet-0 GiB of other components (text encoder, VAE), for _placement
        for d in dirs:
            for f in sorted(os.listdir(d)):
                if f.endswith(".json") and not f.startswith("manifest"):
                    e = json.load(open(os.path.join(d, f)))
                    if e.get("split"):  # split DiT half (runtime_split.py)
                        self._add_split_half(d, e)
                        continue
                    path = os.path.join(d, e["name"] + ".rbln")
                    self.models.setdefault(e["cfg"], []).append(
                        (e["l0"], e["l1"], path, [tuple(x) for x in e["executors"]])
                    )
                    for pm in e.get("prefix_models", []):
                        self.prefix_models.setdefault(e["cfg"], []).append(
                            (
                                pm["family"],
                                pm["l0"],
                                pm["l1"],
                                os.path.join(d, pm["name"] + ".rbln"),
                                [tuple(x) for x in pm["executors"]],
                            )
                        )
                    # chiplet-0 GiB is the binding budget (see UNIFIED); older records only have the total
                    self.sizes[path] = (
                        e["chiplet_gib"][0] if "chiplet_gib" in e else sum(e.get("alloc_gib", {}).values()) / 4
                    )
        for v in list(self.models.values()) + list(self.prefix_models.values()):
            v.sort()
        self.devices = list(device) if isinstance(device, list | tuple) else [device]
        self.device = self.devices[0]
        self.loaded = {}  # cfg -> [runtime...]
        from diffusers.configuration_utils import FrozenDict

        self.config = FrozenDict(config)
        n_layers = max(m[-1][1] for m in self.models.values())
        self.n_layers = n_layers
        self.transformer_blocks = [None] * n_layers
        self.dtype = torch.bfloat16
        self._ctx = "cond"
        self._state = {}
        self.timings: dict[str, list[float]] = {}
        self.load_s = 0.0
        # cfgs whose model runs are verified by repetition (see _run_group): the 2048-class models. The 1024-class flash
        # models never disagreed in 4800 layer runs and every repeated request, so they run once (add "flash" to check them)
        self.verify = {"eager", "xl", "xxl"}
        self.on_oom = []  # callables that free other device memory (e.g. cached VAE graphs) before a cfg switch retries
        for cfg in self.PREFERENCE:  # load the cfg that serves the default 1024 class
            if cfg in self.models:
                self._runtimes(cfg)
                break

    @contextmanager
    def cache_context(self, name):
        prev, self._ctx = self._ctx, name
        try:
            yield
        finally:
            self._ctx = prev

    def _runtimes(self, cfg):
        """Layer-group runtimes of a cfg. Only one cfg is resident: the other one is unloaded first. Room on chiplet 0 is made
        up front (a load that fails half way does not give its memory back, 10-14), cheapest first: `on_oom` hooks."""
        import gc

        import rebel

        if cfg in self.loaded:
            return self.loaded[cfg]
        t0 = time.perf_counter()
        if self.loaded:  # (self._state already describes the request that triggered this load: keep it)
            self.loaded.clear()
            gc.collect()
        place = self._split_placement() if cfg == "xxl" else self._placement(cfg)
        need = {}
        for p, d in place:
            need[d] = need.get(d, 0.0) + self.sizes[p] + 0.3
        for free in [None, *self.on_oom]:
            if free is not None:
                free()
                gc.collect()
            if all(self._chiplet0_free(d) >= n for d, n in need.items()):
                break
        rts = [rebel.Runtime(p, tensor_type="pt", device=dev) for p, dev in place]
        self.load_s += time.perf_counter() - t0
        self.loaded[cfg] = rts
        return rts

    def _local(self, path):
        """Prefix models are read on every request that needs them (a load reads the whole file, mostly Init constants):
        keep a copy on local disk when `local_cache` is set (network storage measured 0.5-1.3 GB/s here)."""
        import shutil

        if not self.local_cache:
            return path
        dst = os.path.join(self.local_cache, os.path.basename(path))
        if not os.path.exists(dst):
            os.makedirs(self.local_cache, exist_ok=True)
            if shutil.disk_usage(self.local_cache).free < os.path.getsize(path) + (50 << 30):
                return path  # shared local disk: keep 50 GB free
            shutil.copyfile(path, dst + ".part")
            os.replace(dst + ".part", dst)
        return dst

    def release(self):
        """Unload the resident DiT (another component needs the device memory); it is loaded again on its next call."""
        import gc

        self.loaded.clear()
        gc.collect()

    @staticmethod
    def _chiplet0_free(device):
        from rebel.device_info import get_device_memory_info

        return get_device_memory_info(device).chiplets[0].free / 2**30

    def _placement(self, cfg):
        """Assign the layer groups of a cfg (in order) to contiguous runs of devices, minimizing the largest chiplet-0 load
        (model chiplet-0 GiB + `self.reserve[device]` for the other components that live on that device)."""
        import itertools

        paths = [p for _, _, p, _ in self.models[cfg]]
        nd = len(self.devices)
        best = None
        for cuts in itertools.combinations_with_replacement(range(len(paths) + 1), nd - 1):
            bounds = [0, *cuts, len(paths)]
            loads = [
                self.reserve.get(dev, 0.0) + sum(self.sizes[p] for p in paths[bounds[i] : bounds[i + 1]])
                for i, dev in enumerate(self.devices)
            ]
            if best is None or max(loads) < best[0]:
                best = (max(loads), bounds)
        bounds = best[1]
        return [(p, self.devices[i]) for i in range(nd) for p in paths[bounds[i] : bounds[i + 1]]]

    # ------------------------------------------------------------------ bucket selection
    def _buckets(self, cfg, kind):
        execs = list(self.models[cfg][0][3]) + [x for pm in self.prefix_models.get(cfg, []) for x in pm[4]]
        return sorted({C for k, C in execs if k == kind})

    def _pick(self, cfg, kind, n):
        for C in self._buckets(cfg, kind):
            if C >= n:
                return C
        return None

    def _run(self, cfg, kind, C, x, extra, rope, pos, mask):
        l0 = self.load_s
        t_start = time.perf_counter()
        rts = self._runtimes(cfg)
        tag = extra[0].float().reshape(1) if kind == "tgt" else torch.zeros(KIND_TAG[kind])
        bt = torch.arange(CONFIGS[cfg].n_part, dtype=torch.int16)
        out = x
        verify = cfg in self.verify
        for rt in rts:
            out = self._run_group(rt, (out, tag, rope, pos, bt, mask), verify)
        self.timings.setdefault("dit_runtime_load", []).append(self.load_s - l0)
        self.timings.setdefault(f"dit_run_{kind}{C}", []).append(time.perf_counter() - t_start)
        return out

    def _run_group(self, rt, args, verify):
        """Run one layer-group model. With `verify`, run it until two results agree (bit for bit) and use that result.

        Why (measured 10-14): models that process 2048-class chunks (16896 tokens, eager or flash attention alike) give a
        different output for identical inputs in ~2-3 % of calls (8-layer models; errors up to half of the output range,
        then spread by attention), on both devices. Single ops and the 1024-class models never did (4800 layer runs), so
        it is not reproducible at op level and looks like a runtime/hardware issue. Re-running is safe: a run rewrites the
        same cache positions with the same values."""

        def once():  # all outputs (split head graphs also return a check of their cache writes)
            o = rt.run(*args)
            return [x.clone() for x in o] if isinstance(o, list | tuple) else [o.clone()]

        first = once()
        if not verify:
            return first[0]
        seen = [first]
        while True:
            o = once()
            for prev in seen:
                if all(torch.equal(a, b) for a, b in zip(o, prev, strict=True)):
                    if len(seen) > 1:
                        self.timings.setdefault("dit_verify_mismatch", []).append(float(len(seen) - 1))
                    return o[0]
            seen.append(o)
            if len(seen) > 6:
                raise RuntimeError("DiT model output does not repeat over 7 runs")

    def _tick(self, k, t0):
        self.timings.setdefault(k, []).append(time.perf_counter() - t0)

    # ------------------------------------------------------------------ prefix
    def _cfg_for(self, n_tgt, segs, n_ctx=1):
        """The first cfg in PREFERENCE that has a target bucket and room for the prefix next to the target region."""
        for cfg in self.PREFERENCE:
            if cfg not in self.models:
                continue
            C_t = self._pick(cfg, "tgt", n_tgt)
            if C_t is None:
                continue
            if self._prefix_end(cfg, segs, 0, C_t) is not None:
                return cfg
        raise ValueError(
            f"no compiled cache can hold {n_tgt} target tokens plus this prefix ({sum(s.length for s in segs)} tokens in all)"
        )

    def _prefix(self, cfg, segs, enc, txt_valid, cond_lat, lay, mask):
        """Write the prefix chunks (text, condition images) to the cache. enc [Lt, 4096] (text rows, joint order), txt_valid
        [Lt] (False for prompt padding of a batch, kept out of the mask), cond_lat [Nc, 64]. Each chunk sees the mask of
        the chunks up to itself (later positions may still hold another request's keys)."""
        chunks = []  # (kind, C, x, rope, pos, mask snapshot)
        t_ptr = i_ptr = 0
        for s in segs:
            if s.kind == "tgt":
                continue
            if s.kind == "txt":
                rows, valid = enc[t_ptr : t_ptr + s.length], txt_valid[t_ptr : t_ptr + s.length]
                t_ptr += s.length
                big = max(self._buckets(cfg, "txt"))
                for o in range(0, s.length, big):  # long text is causal, so simply split it into multiple chunks
                    n = min(big, s.length - o)
                    C = self._pick(cfg, "txt", n)
                    chunks.append(
                        self._chunk(
                            cfg, "txt", C, rows[o : o + n], s.rope_rows[:, o : o + n], n, lay, mask, valid[o : o + n]
                        )
                    )
            else:
                rows = cond_lat[i_ptr : i_ptr + s.length]
                i_ptr += s.length
                C = self._pick(cfg, "img", s.length)
                if C is None:
                    raise ValueError(f"no bucket can hold {s.length} condition image tokens ({cfg})")
                chunks.append(self._chunk(cfg, "img", C, rows, s.rope_rows, s.length, lay, mask))
        t0 = time.perf_counter()
        resident = set(self.models[cfg][0][3])
        if all((k, C) in resident for k, C, *_ in chunks):
            for kind, C, x, rope, pos, m in chunks:
                self._run(cfg, kind, C, x, [], rope, pos, m)
        else:
            self._prefix_layered(cfg, chunks)
        self._tick("dit_prefix", t0)

    @staticmethod
    def _chunk(cfg, kind, C, rows, rope_rows, n, lay, mask, valid=None):
        start = lay.place(C)
        x = torch.zeros(C, rows.shape[-1])
        x[:n] = rows.float()
        rope = torch.zeros(3, C, dtype=torch.long)
        rope[:, :n] = rope_rows
        mask[0, start : start + n] = 1 if valid is None else valid.float()
        return kind, C, x, rope, (start + torch.arange(C, dtype=torch.int32))[None], mask.clone()

    def _prefix_layered(self, cfg, chunks):
        """Prefix through models loaded for this phase only: layer range by layer range, every chunk in order (chunk i at
        layer l only needs the cache of chunks < i at layer l). Each model goes to the device of the resident model that
        holds its layers (the caches live there)."""
        import gc

        import rebel

        self._runtimes(cfg)  # resident models (they own the caches) first
        need = {(k, C) for k, C, *_ in chunks}  # the first prefix-model family that has every chunk's executor
        family = next(
            f
            for f in sorted({pm[0] for pm in self.prefix_models[cfg]})
            if need <= {tuple(x) for pm in self.prefix_models[cfg] if pm[0] == f for x in pm[4]}
        )
        dev = {}
        for (l0, l1, *_), (_, d) in zip(self.models[cfg], self._placement(cfg), strict=True):
            for li in range(l0, l1):
                dev[li] = d
        bt = torch.arange(CONFIGS[cfg].n_part, dtype=torch.int16)
        hs = [x for _, _, x, *_ in chunks]
        verify = cfg in self.verify
        for fam, a, _, path, _ in self.prefix_models[cfg]:
            if fam != family:
                continue
            t0 = time.perf_counter()
            rt = rebel.Runtime(self._local(path), tensor_type="pt", device=dev[a])
            self.timings.setdefault("dit_prefix_load", []).append(time.perf_counter() - t0)
            for i, (kind, _, _, rope, pos, m) in enumerate(chunks):
                hs[i] = self._run_group(rt, (hs[i], torch.zeros(KIND_TAG[kind]), rope, pos, bt, m), verify)
            del rt
            gc.collect()

    # ------------------------------------------------------------------ pipeline call
    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        img_shapes,
        img_mask,
        encoder_hidden_states_mask=None,
        attention_kwargs=None,
        kv_cache=None,
        kv_cache_mode=None,
        return_dict=False,
    ):
        """Same inputs/outputs as the original transformer. Samples of a batch run one after the other, each with its own
        cache region; like the original, the joint layout (text padding included) comes from row 0 of `img_mask`, and
        padded text rows are excluded through the cache mask."""
        B = hidden_states.shape[0]
        repeats = torch.where(img_mask, 4, 1)[0]
        pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        n_img = int(pad_mask.sum())
        if n_img != hidden_states.shape[1]:  # e.g. an odd patch grid: the original fails the same way
            raise RuntimeError(
                f"shape mismatch: {hidden_states.shape[1]} image tokens for {n_img} image slots (as the original transformer)"
            )
        segs = build_segments(pad_mask, img_shapes[0])
        N = segs[-1].length
        if self._ctx == "cond" and kv_cache_mode in (
            None,
            "extract",
        ):  # a new request starts with its cond extract call
            cfg = self._cfg_for(N, segs)
            self._state = {"cfg": cfg, "cursor": 0, "last_prefix_end": 0, "all_volatile": False, "ctx": {}}
        if not self._state:
            raise RuntimeError("the first call of a request must be the cond context with kv_cache_mode extract/None")
        ts = torch.as_tensor(timestep).reshape(-1)
        outs = []
        for b in range(B):
            enc_mask = None if encoder_hidden_states_mask is None else encoder_hidden_states_mask[b]
            outs.append(
                self._call_one(
                    (self._ctx, b),
                    hidden_states[b],
                    ts[b if ts.numel() > 1 else 0],
                    encoder_hidden_states[b],
                    enc_mask,
                    img_mask[0],
                    segs,
                    kv_cache_mode,
                )
            )
        return (torch.stack(outs).to(torch.bfloat16),)

    def _tgt_start(self, cfg, C_t, prefix_end):
        if cfg == "xxl":
            return self._split_slots(prefix_end)[0]
        """Where the target goes: right after the prefixes (attention cost grows with the position, since a chunk attends
        to cache positions before its end). Its K/V are rewritten on every call, so it can move as contexts are added.
        Raises ValueError if it does not fit."""
        lay = CacheLayout(CONFIGS[cfg])
        lay.cursor = prefix_end
        return lay.place(C_t)

    def _prefix_end(self, cfg, segs, start, C_t):
        """Simulate placing the prefix pieces from `start`; returns their end, or None if they (plus the target) do not fit."""
        lay = CacheLayout(CONFIGS[cfg])
        lay.cursor = start
        try:
            for s in segs:
                if s.kind == "txt":
                    big = max(self._buckets(cfg, "txt"))
                    for o in range(0, s.length, big):
                        lay.place(self._pick(cfg, "txt", min(big, s.length - o)))
                elif s.kind == "img" and cfg == "xxl" and s.length > self._small_img:
                    if s.length > 2 * self.split_P:
                        raise ValueError(f"condition image of {s.length} tokens > {2 * self.split_P}")
                    self._place_pieces(lay)
                elif s.kind == "img":
                    lay.place(self._pick(cfg, "img", s.length))
            end = lay.cursor
            self._tgt_start(cfg, C_t, end)
        except (TypeError, ValueError):
            return None
        return end

    def _call_one(self, key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode):
        st_all = self._state
        cfg = st_all["cfg"]
        if cfg == "xxl":
            return self._call_one_split(key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode)
        tgt = segs[-1]
        N = tgt.length
        n_cond = hs.shape[0] - N
        cond_lat, tgt_lat = hs[:n_cond], hs[n_cond:]
        # original: image slot rows of encoder_hidden_states are replaced by latents, so only text rows go into txt_in
        slot = layout[: enc_all.shape[0]]
        enc = enc_all[~slot]
        txt_valid = torch.ones(enc.shape[0], dtype=torch.bool) if enc_mask is None else enc_mask[~slot].bool()
        C_t = self._pick(cfg, "tgt", N)
        ctxs = st_all["ctx"]
        st = ctxs.get(key)
        if st is None:
            st = {"volatile": True, "start": 0}
            if not st_all["all_volatile"]:
                end = self._prefix_end(cfg, segs, st_all["cursor"], C_t)
                if end is not None:  # a dedicated prefix region, kept for the whole request
                    st = {"volatile": False, "start": st_all["cursor"]}
                    st_all["cursor"] = end
                else:  # out of room: from now on every context rewrites its prefix from 0 on every call
                    st_all["all_volatile"] = True
            ctxs[key] = st
        if st_all["all_volatile"]:
            st["volatile"], st["start"] = True, 0
        if st.get("mask") is None or kv_cache_mode is None or st["volatile"]:
            lay = CacheLayout(CONFIGS[cfg])
            lay.cursor = st["start"]
            mask = torch.zeros(1, CONFIGS[cfg].cap)
            self._prefix(cfg, segs, enc, txt_valid, cond_lat, lay, mask)
            st["mask"], st["end"] = mask, lay.cursor
        # shared target region after every dedicated prefix (or after this context's own prefix when all are volatile)
        ts_ = self._tgt_start(cfg, C_t, st["end"] if st_all["all_volatile"] else st_all["cursor"])
        mask = st["mask"].clone()
        mask[0, ts_ : ts_ + N] = 1
        x = torch.zeros(C_t, LATENT_C)
        x[:N] = tgt_lat.float()
        rope = torch.zeros(3, C_t, dtype=torch.long)
        rope[:, :N] = tgt.rope_rows
        t0 = time.perf_counter()
        out = self._run(
            cfg, "tgt", C_t, x, [t.float().reshape(1)], rope, (ts_ + torch.arange(C_t, dtype=torch.int32))[None], mask
        )
        self._tick("dit_target", t0)
        return out[:N]

    # ------------------------------------------------------------------ xxl: split blocks (dit_split.py, runtime_split.py)
    _small_img = 4352  # xxl: condition images up to this many tokens are one chunk (dual models), larger ones split

    def _add_split_half(self, d, e):
        from .runtime_split import DUAL_EXECUTORS

        files = []
        for fe in e["files"]:
            path = os.path.join(d, fe["name"] + ".rbln")
            self.sizes[path] = fe["chiplet_gib"][0]
            files.append((fe, path))
        self.split_halves.append((e["l0"], e["l1"], files))
        self.split_halves.sort(key=lambda h: h[0])
        self.split_P = e["P"]
        CONFIGS["xxl"] = KVConfig(**e["kv"])
        execs = [tuple(x) for x in DUAL_EXECUTORS] + [("img", 2 * e["P"]), ("tgt", 2 * e["P"])]
        self.models["xxl"] = [(a, b, None, execs) for a, b, _ in self.split_halves]

    def _split_placement(self):
        """Half i of the split bank goes to device i (wrapping around): a half's files share weights only with each other.
        Dual (prefix) models are not resident: _prefix_split loads them one at a time."""
        return [
            (p, self.devices[i % len(self.devices)])
            for i, (_, _, files) in enumerate(self.split_halves)
            for fe, p in files
            if fe["role"] != "dual"
        ]

    def _split_rts(self):
        rts = iter(self._runtimes("xxl"))
        heads, steps, mods = {}, {}, []
        for _, _, files in self.split_halves:
            for fe, _ in files:
                if fe["role"] == "dual":
                    continue
                rt = next(rts)
                if fe["role"] == "mod":
                    mods.append(rt)
                elif fe["role"] == "head":
                    heads[fe["layer"]] = rt
                else:
                    steps[fe["layer"]] = rt
        return heads, steps, mods

    def _split_slots(self, start):
        """Cache slots s1 < s2 (one piece each, each inside one partition) for a split block placed from `start`;
        ValueError if they do not fit."""
        lay = CacheLayout(CONFIGS["xxl"])
        lay.cursor = start
        return self._place_pieces(lay) + (lay.cursor,)

    def _place_pieces(self, lay):
        s1 = lay.place(self.split_P)
        s2 = lay.place(self.split_P)
        return s1, s2

    def _split_inputs(self, rows, rope_rows, s1, s2, mask):
        """A split block of N rows → graph inputs (2P rows: A = first ceil(N/2) rows, B = the rest, each padded to P),
        slot positions, and the X/Y cache masks: X holds A at s1 and B at s2, Y holds B at s1 and A at s2."""
        P = self.split_P
        N = rows.shape[0]
        nA = (N + 1) // 2
        nB = N - nA
        x = torch.zeros(2 * P, rows.shape[-1])
        x[:nA], x[P : P + nB] = rows[:nA].float(), rows[nA:].float()
        rope = torch.zeros(3, 2 * P, dtype=torch.long)
        rope[:, :nA], rope[:, P : P + nB] = rope_rows[:, :nA], rope_rows[:, nA:]
        pos = torch.stack([s1 + torch.arange(P, dtype=torch.int32), s2 + torch.arange(P, dtype=torch.int32)])
        mask[0, s1 : s1 + nA] = 1
        mask[0, s2 : s2 + nB] = 1
        mask[1, s1 : s1 + nB] = 1
        mask[1, s2 : s2 + nA] = 1
        return x, rope, pos, (nA, nB)

    @staticmethod
    def _dual_inputs(C, rows, rope_rows, lay, mask, valid=None):
        """A chunk written to both caches at the same positions (both mask rows)."""
        n = rows.shape[0]
        start = lay.place(C)
        x = torch.zeros(C, rows.shape[-1])
        x[:n] = rows.float()
        rope = torch.zeros(3, C, dtype=torch.long)
        rope[:, :n] = rope_rows
        mask[:, start : start + n] = 1 if valid is None else valid.float()
        return x, rope, (start + torch.arange(C, dtype=torch.int32))[None]

    def _split_mod(self, mod_rt, t):
        """(modulation [1, 4D], output-norm scale [1, D]) of timestep t, from a half's mod graph (verified like the rest)."""
        o = mod_rt.run(t.float().reshape(1))
        return o[0].clone(), o[1].clone()

    def _split_layer(self, li, h, mod, nscale, rope, pos, mask):
        """Layer li of a split block: the head graph writes A/B to slot 1 of X/Y, the step graph computes the layer
        (A and B at slot 2)."""
        heads, steps, _ = self._split_rts()
        bt = torch.arange(CONFIGS["xxl"].n_part, dtype=torch.int16)
        part = CONFIGS["xxl"].part
        bt1 = (pos[0, :1] // part).to(torch.int16)  # slot 1 lies in one partition: the head sees only that one
        pos0 = pos[0:1] % part
        own = torch.ones(1, CONFIGS["xxl"].part)
        hh = self._run_group(heads[li], (h, mod, rope, pos0, bt1, own), True)
        if li == 0:  # the layer-0 head applies img_in to the latents
            h = hh
        extra = [nscale] if li + 1 == self.n_layers else []
        return self._run_group(steps[li], (h, mod, *extra, rope, pos[1:2], bt, mask), True)

    def _split_chain(self, x, t, rope, pos, mask):
        _, _, mods = self._split_rts()
        h = x
        for (l0, l1, _), mod_rt in zip(self.split_halves, mods, strict=True):
            mod, nscale = self._split_mod(mod_rt, t)
            for li in range(l0, l1):
                h = self._split_layer(li, h, mod, nscale, rope, pos, mask)
        return h

    def _prefix_split(self, segs, enc, txt_valid, cond_lat, lay, mask):
        """Write the prefix layer range by layer range (chunk i at layer l only needs the caches of chunks < i at layer l),
        so that one dual model at a time is loaded next to the resident split models."""
        import gc

        import rebel

        t0 = time.perf_counter()
        chunks = []  # [kind, x, rope, pos, mask snapshot]; kind "split" for blocks above one piece
        t_ptr = i_ptr = 0
        for s in segs:
            if s.kind == "tgt":
                continue
            if s.kind == "txt":
                rows, valid = enc[t_ptr : t_ptr + s.length], txt_valid[t_ptr : t_ptr + s.length]
                t_ptr += s.length
                big = max(self._buckets("xxl", "txt"))
                for o in range(0, s.length, big):
                    n = min(big, s.length - o)
                    C = self._pick("xxl", "txt", n)
                    x, rope, pos = self._dual_inputs(
                        C, rows[o : o + n], s.rope_rows[:, o : o + n], lay, mask, valid[o : o + n]
                    )
                    chunks.append(["txt", x, rope, pos, mask.clone()])
                continue
            rows = cond_lat[i_ptr : i_ptr + s.length]
            i_ptr += s.length
            if s.length <= self._small_img:
                x, rope, pos = self._dual_inputs(self._small_img, rows, s.rope_rows, lay, mask)
                chunks.append(["img", x, rope, pos, mask.clone()])
            else:
                s1, s2 = self._place_pieces(lay)
                x, rope, pos, _ = self._split_inputs(rows, s.rope_rows, s1, s2, mask)
                chunks.append(["split", x, rope, pos, mask.clone()])
        _, _, mods = self._split_rts()
        bt = torch.arange(CONFIGS["xxl"].n_part, dtype=torch.int16)
        need = {c[0] for c in chunks} - {"split"}  # dual families this prefix needs ("txt", "img")
        for hi, ((_, _, files), mod_rt) in enumerate(zip(self.split_halves, mods, strict=True)):
            mod, nscale = self._split_mod(mod_rt, torch.zeros(1))  # prefix: t = 0
            dev = self.devices[hi % len(self.devices)]
            ranges = sorted({(fe["l0"], fe["l1"]) for fe, _ in files if fe["role"] == "dual"})
            for a, b in ranges:
                rts = {}
                for fe, path in files:
                    if fe["role"] != "dual" or (fe["l0"], fe["l1"]) != (a, b) or fe["family"] not in need:
                        continue
                    for free in [None, *self.on_oom]:
                        if free is not None:
                            free()
                            gc.collect()
                        if self._chiplet0_free(dev) >= self.sizes[path] + 0.2:
                            break
                    t1 = time.perf_counter()
                    rts[fe["family"]] = rebel.Runtime(self._local(path), tensor_type="pt", device=dev)
                    self.timings.setdefault("dit_prefix_load", []).append(time.perf_counter() - t1)
                for c in chunks:
                    kind, h, rope, pos, m = c
                    if kind == "split":
                        for li in range(a, b):
                            h = self._split_layer(li, h, mod, nscale, rope, pos, m)
                    else:
                        h = self._run_group(rts[kind], (h, torch.zeros(KIND_TAG[kind]), mod, rope, pos, bt, m), True)
                    c[1] = h
                del rts
                gc.collect()
        self._tick("dit_prefix", t0)

    def _call_one_split(self, key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode):
        """_call_one for xxl: same context handling (dedicated prefix regions, shared target slots after them)."""
        st_all = self._state
        tgt = segs[-1]
        N = tgt.length
        n_cond = hs.shape[0] - N
        cond_lat, tgt_lat = hs[:n_cond], hs[n_cond:]
        slot = layout[: enc_all.shape[0]]
        enc = enc_all[~slot]
        txt_valid = torch.ones(enc.shape[0], dtype=torch.bool) if enc_mask is None else enc_mask[~slot].bool()
        C_t = 2 * self.split_P
        ctxs = st_all["ctx"]
        st = ctxs.get(key)
        if st is None:
            st = {"volatile": True, "start": 0}
            if not st_all["all_volatile"]:
                end = self._prefix_end("xxl", segs, st_all["cursor"], C_t)
                if end is not None:
                    st = {"volatile": False, "start": st_all["cursor"]}
                    st_all["cursor"] = end
                else:
                    st_all["all_volatile"] = True
            ctxs[key] = st
        if st_all["all_volatile"]:
            st["volatile"], st["start"] = True, 0
        cap = CONFIGS["xxl"].cap
        if st.get("mask") is None or kv_cache_mode is None or st["volatile"]:
            lay = CacheLayout(CONFIGS["xxl"])
            lay.cursor = st["start"]
            mask = torch.zeros(2, cap)
            self._prefix_split(segs, enc, txt_valid, cond_lat, lay, mask)
            st["mask"], st["end"] = mask, lay.cursor
        s1, s2, _ = self._split_slots(st["end"] if st_all["all_volatile"] else st_all["cursor"])
        mask = st["mask"].clone()
        x, rope, pos, (nA, nB) = self._split_inputs(tgt_lat, tgt.rope_rows, s1, s2, mask)
        t0 = time.perf_counter()
        out = self._split_chain(x, t.float().reshape(1), rope, pos, mask)
        self._tick("dit_target", t0)
        P = self.split_P
        return torch.cat([out[:nA], out[P : P + nB]])


def compile_unified(
    model,
    out_dir: str,
    cfg_name: str,
    l0: int,
    l1: int,
    npu="RBLN-CR13",
    log=print,
    num_layers=None,
    executors=None,
):
    """Layers [l0, l1) of one cfg as a single compiled model with multiple (kind, bucket) executors, plus the cfg's
    PREFIX_MODELS for the same layers, all in one compile context (so they share caches and weights). Models with different
    layer ranges share nothing, so they can be compiled concurrently in separate processes."""
    import rebel
    from rebel.compile_context import CompileContext

    nl = num_layers or NUM_LAYERS
    cfg = CONFIGS[cfg_name]
    execs = executors or UNIFIED[cfg_name][0]
    os.makedirs(out_dir, exist_ok=True)
    name = f"{cfg_name}_L{l0}-{l1}"
    path = os.path.join(out_dir, name + ".rbln")
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        return
    ctx = CompileContext(use_weight_sharing=True)
    caches = {}
    for li in range(l0, l1):
        for nm in cache_names(cfg_name, li):
            t = torch.zeros(cfg.n_part, cfg.heads, cfg.part, cfg.head_dim)
            ctx.mark_static_address(t, nm)
            caches[nm] = (li, t)
    shared = DiTKVShared(model)

    def compile_model(a, b, exs_spec, out_path):
        mod = DiTKVUnified(shared, a, b, nl, cfg).eval()
        layer_caches = [(nm, t) for nm, (li, t) in caches.items() if a <= li < b]
        infos, exs = [], []
        for kind, C in exs_spec:
            xd = (HIDDEN if kind == "txt" else LATENT_C) if a == 0 else HIDDEN
            info = [
                ("x", [C, xd], "float32"),
                ("kind_tag", [KIND_TAG[kind]], "float32"),
                ("rope_idx", [3, C], "int64"),
                ("cache_pos", [1, C], "int32"),
                ("block_tables", [cfg.n_part], "int16"),
                ("mask", [1, cfg.cap], "float32"),
            ]
            ex = [
                torch.zeros(C, xd),
                torch.full((KIND_TAG[kind],), 0.5),
                torch.zeros(3, C, dtype=torch.long),
                torch.arange(C, dtype=torch.int32)[None],
                torch.arange(cfg.n_part, dtype=torch.int16),
                torch.ones(1, cfg.cap),
            ]
            for n, t in layer_caches:
                info.append((n, list(t.shape), "float32"))
                ex.append(t)
            infos.append(info)
            exs.append(ex)
        if len(infos) == 1:  # a single executor is passed unwrapped (a one-element list is not accepted)
            infos, exs = infos[0], exs[0]
        t0 = time.perf_counter()
        c = rebel.compile_from_torch(mod, input_info=infos, example_inputs=exs, compile_context=ctx, npu=npu)
        c.save(out_path)
        by, c_by = c.get_alloc_per_node_by_key(), c.get_alloc_per_chiplet_by_key()
        return {
            "compile_s": round(time.perf_counter() - t0, 1),
            "alloc_gib": {k: round(sum(v) / 2**30, 2) for k, v in by.items() if sum(v) > 2**26},
            # device DRAM is partitioned per chiplet and chiplet 0 receives most of the scratch: it is the binding budget
            "chiplet_gib": [round(sum(v[0][ch] for v in c_by.values()) / 2**30, 2) for ch in range(4)],
        }

    entry = {"name": name, "cfg": cfg_name, "l0": l0, "l1": l1, "executors": [list(e) for e in execs]}
    entry.update(compile_model(l0, l1, execs, path))
    log(f"compiled {name} in {entry['compile_s']:.0f}s alloc={entry['alloc_gib']} chiplets={entry['chiplet_gib']}")
    prefix = []
    for k, (pex, per) in enumerate(PREFIX_MODELS.get(cfg_name, [])):
        for a in range(l0, l1, per):
            b = min(a + per, l1)
            pname = f"{name}.pfx{k}_L{a}-{b}"
            e = {"name": pname, "family": k, "l0": a, "l1": b, "executors": [list(x) for x in pex]}
            e.update(compile_model(a, b, pex, os.path.join(out_dir, pname + ".rbln")))
            log(f"compiled {pname} in {e['compile_s']:.0f}s chiplets={e['chiplet_gib']}")
            prefix.append(e)
    if prefix:
        entry["prefix_models"] = prefix
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(entry, f, indent=2)
