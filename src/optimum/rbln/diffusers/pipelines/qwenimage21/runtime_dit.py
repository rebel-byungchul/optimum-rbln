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
    PREFERENCE = ("flash", "eager")

    def __init__(self, bank_dir, config: dict, device: int = 0, reserve: dict | None = None):
        dirs = [bank_dir] if isinstance(bank_dir, str) else list(bank_dir)
        self.bank_dir = dirs[0]
        self.models = {}  # cfg -> [(l0, l1, path, executors)]
        self.sizes = {}  # path -> chiplet-0 memory (GiB, from the compile record)
        self.reserve = dict(
            reserve or {}
        )  # device -> chiplet-0 GiB of other components (text encoder, VAE), for _placement
        for d in dirs:
            for f in sorted(os.listdir(d)):
                if f.endswith(".json") and not f.startswith("manifest"):
                    e = json.load(open(os.path.join(d, f)))
                    path = os.path.join(d, e["name"] + ".rbln")
                    self.models.setdefault(e["cfg"], []).append(
                        (e["l0"], e["l1"], path, [tuple(x) for x in e["executors"]])
                    )
                    # chiplet-0 GiB is the binding budget (see UNIFIED); older records only have the total
                    self.sizes[path] = (
                        e["chiplet_gib"][0] if "chiplet_gib" in e else sum(e.get("alloc_gib", {}).values()) / 4
                    )
        for v in self.models.values():
            v.sort()
        self.devices = list(device) if isinstance(device, list | tuple) else [device]
        self.device = self.devices[0]
        self.loaded = {}  # cfg -> [runtime...]
        from diffusers.configuration_utils import FrozenDict

        self.config = FrozenDict(config)
        n_layers = max(m[-1][1] for m in self.models.values())
        self.transformer_blocks = [None] * n_layers
        self.dtype = torch.bfloat16
        self._ctx = "cond"
        self._state = {}
        self.timings: dict[str, list[float]] = {}
        self.load_s = 0.0
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
        """Layer-group runtimes for a cfg. If device memory runs out, unload the runtimes of other cfgs and retry
        (a request uses only one cfg, so the cache contents of other cfgs can be discarded)."""
        import gc

        import rebel

        if cfg in self.loaded:
            return self.loaded[cfg]
        t0 = time.perf_counter()
        if (
            self.loaded
        ):  # only one cfg fits next to the other components: free first (a failed partial load leaks memory)
            self.loaded.clear()  # (self._state already describes the request that triggered this load: keep it)
            for free in self.on_oom:
                free()
            gc.collect()
        while True:
            try:
                rts = [rebel.Runtime(p, tensor_type="pt", device=dev) for p, dev in self._placement(cfg)]
                break
            except RuntimeError as e:
                others = [k for k in self.loaded if k != cfg]
                oom = any(
                    k in str(e) for k in ("EBUSY", "SYS_ERROR", "device memory")
                )  # -14 also means a full chiplet
                if not oom or not (others or self.on_oom):
                    raise
                if not others:  # already freed everything we can: stop after one retry
                    if getattr(self, "_retried", False):
                        raise
                    self._retried = True
            # outside the except block, so the partly created runtimes held by the traceback are released first
            for k in others:
                del self.loaded[k]
            for free in self.on_oom:
                free()
            gc.collect()
        self._retried = False
        self.load_s += time.perf_counter() - t0
        self.loaded[cfg] = rts
        return rts

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
        return sorted({C for k, C in self.models[cfg][0][3] if k == kind})

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
        for rt in rts:
            out = rt.run(out, tag, rope, pos, bt, mask)
            out = out[0] if isinstance(out, list | tuple) else out
        self.timings.setdefault("dit_runtime_load", []).append(self.load_s - l0)
        self.timings.setdefault(f"dit_run_{kind}{C}", []).append(time.perf_counter() - t_start)
        return out

    def _tick(self, k, t0):
        self.timings.setdefault(k, []).append(time.perf_counter() - t0)

    # ------------------------------------------------------------------ prefix
    def _cfg_for(self, n_tgt, segs, n_ctx=1):
        """The first cfg in PREFERENCE that has a target bucket and room for prefix + target."""
        for cfg in self.PREFERENCE:
            if cfg not in self.models:
                continue
            C_t = self._pick(cfg, "tgt", n_tgt)
            if C_t is None:
                continue
            try:
                need = self._prefix_need(cfg, segs)
            except (TypeError, ValueError):
                continue
            if need + C_t <= CONFIGS[cfg].cap:  # (CFG's second context falls back to a volatile region if needed)
                return cfg
        raise ValueError(
            f"no compiled cache can hold {n_tgt} target tokens plus this prefix: condition images must be ≤ 4352 tokens "
            "(output_resolution ≤ 1024 class); a 2048-class output takes up to 3 condition images (eager cache < 32768); "
            "more images need a 1024-class output (flash cache, up to 10 images)"
        )

    def _prefix(self, cfg, segs, enc, cond_lat, lay, mask):
        """Write the prefix chunks (text, condition images) to the cache. enc [Lt, 4096] (text rows, joint order), cond_lat [Nc, 64]."""
        t_ptr = i_ptr = 0
        for s in segs:
            if s.kind == "tgt":
                continue
            if s.kind == "txt":
                rows = enc[t_ptr : t_ptr + s.length]
                t_ptr += s.length
                big = max(self._buckets(cfg, "txt"))
                for o in range(0, s.length, big):  # long text is causal, so simply split it into multiple chunks
                    n = min(big, s.length - o)
                    C = self._pick(cfg, "txt", n)
                    self._write_chunk(cfg, "txt", C, rows[o : o + n], s.rope_rows[:, o : o + n], n, lay, mask)
            else:
                rows = cond_lat[i_ptr : i_ptr + s.length]
                i_ptr += s.length
                C = self._pick(cfg, "img", s.length)
                if C is None:
                    raise ValueError(f"no bucket can hold {s.length} condition image tokens ({cfg})")
                self._write_chunk(cfg, "img", C, rows, s.rope_rows, s.length, lay, mask)

    def _prefix_need(self, cfg, segs):
        """Cache length needed to write the prefix (simulates placement with the same rules)."""
        lay = CacheLayout(CONFIGS[cfg])
        for s in segs:
            if s.kind == "txt":
                big = max(self._buckets(cfg, "txt"))
                for o in range(0, s.length, big):
                    lay.place(self._pick(cfg, "txt", min(big, s.length - o)))
            elif s.kind == "img":
                lay.place(self._pick(cfg, "img", s.length))
        return lay.cursor

    def _write_chunk(self, cfg, kind, C, rows, rope_rows, n, lay, mask):
        start = lay.place(C)
        x = torch.zeros(C, rows.shape[-1])
        x[:n] = rows.float()
        rope = torch.zeros(3, C, dtype=torch.long)
        rope[:, :n] = rope_rows
        mask[0, start : start + n] = 1
        t0 = time.perf_counter()
        self._run(cfg, kind, C, x, [], rope, (start + torch.arange(C, dtype=torch.int32))[None], mask)
        self._tick(f"dit_prefix_{kind}", t0)

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
        assert hidden_states.shape[0] == 1, "only batch 1 is supported"
        if encoder_hidden_states_mask is not None and not bool(encoder_hidden_states_mask.all()):
            raise NotImplementedError("batch 1 has no prompt padding")
        repeats = torch.where(img_mask, 4, 1)[0]
        pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        segs = build_segments(pad_mask, img_shapes[0])
        tgt = segs[-1]
        N = tgt.length
        n_cond = hidden_states.shape[1] - N
        cond_lat, tgt_lat = hidden_states[0, :n_cond], hidden_states[0, n_cond:]
        # original: image slot rows of encoder_hidden_states are replaced by latents, so only text rows go into txt_in
        enc_all = encoder_hidden_states[0]
        slot = img_mask[0][: enc_all.shape[0]]
        enc = enc_all[~slot]
        if self._ctx == "cond" and kv_cache_mode in (None, "extract"):
            self._req_cfg = self._cfg_for(N, segs)
        cfg = self._req_cfg
        C_t = self._pick(cfg, "tgt", N)

        # new request: if the cond context arrives as extract (or without cache), restart placement from scratch
        if self._ctx == "cond" and kv_cache_mode in (None, "extract"):
            self._state = {"cfg": cfg, "end": 0, "ctx": {}}
        if self._state.get("cfg") != cfg:
            raise RuntimeError("cond/uncond use different cache modes within one request")
        ctxs = self._state["ctx"]
        st = ctxs.get(self._ctx)
        rerun = st is None or kv_cache_mode is None or st["volatile"]
        if rerun:
            if st is None:
                need = self._prefix_need(cfg, segs)
                lay = CacheLayout(CONFIGS[cfg])
                lay.cursor = self._state["end"]
                volatile = False
                try:  # region dedicated to this context: [prefix][target]
                    for _ in range(1):
                        probe = CacheLayout(CONFIGS[cfg])
                        probe.cursor = self._state["end"]
                        probe.cursor += need
                        probe.place(C_t)
                except ValueError:
                    volatile = True
                if volatile:  # out of capacity: rewrite right after the cond prefix (overlapping the cond target region) on every call
                    base = ctxs["cond"]["prefix_end"] if "cond" in ctxs else 0
                    lay = CacheLayout(CONFIGS[cfg])
                    lay.cursor = base
                st = {"start": lay.cursor, "volatile": volatile}
                ctxs[self._ctx] = st
            lay = CacheLayout(CONFIGS[cfg])
            lay.cursor = st["start"]
            mask = torch.zeros(1, CONFIGS[cfg].cap)
            self._prefix(cfg, segs, enc, cond_lat, lay, mask)
            st["prefix_end"] = lay.cursor
            st["tgt_start"] = lay.place(C_t)
            st["mask"] = mask
            if not st["volatile"]:
                self._state["end"] = max(self._state["end"], lay.cursor)
        mask = st["mask"].clone()
        ts = st["tgt_start"]
        mask[0, ts : ts + N] = 1
        x = torch.zeros(C_t, LATENT_C)
        x[:N] = tgt_lat.float()
        rope = torch.zeros(3, C_t, dtype=torch.long)
        rope[:, :N] = tgt.rope_rows
        t0 = time.perf_counter()
        out = self._run(
            cfg,
            "tgt",
            C_t,
            x,
            [timestep.float().reshape(1)],
            rope,
            (ts + torch.arange(C_t, dtype=torch.int32))[None],
            mask,
        )
        self._tick("dit_target", t0)
        return (out[:N][None].to(torch.bfloat16),)


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
    """Layers [l0, l1) of one cfg as a single compiled model with multiple (kind, bucket) executors. Weights are shared only
    within the same process, so models with different layer ranges can be compiled concurrently in separate processes."""
    import rebel
    from rebel.compile_context import CompileContext

    nl = num_layers or NUM_LAYERS
    cfg = CONFIGS[cfg_name]
    execs = executors or UNIFIED[cfg_name][0]
    os.makedirs(out_dir, exist_ok=True)
    name = f"{cfg_name}_L{l0}-{l1}"
    path = os.path.join(out_dir, name + ".rbln")
    if os.path.exists(path):
        return
    ctx = CompileContext(use_weight_sharing=True)
    caches = []
    for li in range(l0, l1):
        for nm in cache_names(cfg_name, li):
            t = torch.zeros(cfg.n_part, cfg.heads, cfg.part, cfg.head_dim)
            ctx.mark_static_address(t, nm)
            caches.append((nm, t))
    mod = DiTKVUnified(DiTKVShared(model), l0, l1, nl, cfg).eval()
    infos, exs = [], []
    for kind, C in execs:
        xd = (HIDDEN if kind == "txt" else LATENT_C) if l0 == 0 else HIDDEN
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
        for n, t in caches:
            info.append((n, list(t.shape), "float32"))
            ex.append(t)
        infos.append(info)
        exs.append(ex)
    t0 = time.perf_counter()
    c = rebel.compile_from_torch(mod, input_info=infos, example_inputs=exs, compile_context=ctx, npu=npu)
    c.save(path)
    by = c.get_alloc_per_node_by_key()
    c_by = c.get_alloc_per_chiplet_by_key()
    entry = {
        "name": name,
        "cfg": cfg_name,
        "l0": l0,
        "l1": l1,
        "executors": [list(e) for e in execs],
        "compile_s": round(time.perf_counter() - t0, 1),
        "alloc_gib": {k: round(sum(v) / 2**30, 2) for k, v in by.items() if sum(v) > 2**26},
        # device DRAM is partitioned per chiplet and chiplet 0 receives most of the scratch: it is the binding budget
        "chiplet_gib": [round(sum(v[0][c] for v in c_by.values()) / 2**30, 2) for c in range(4)],
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(entry, f, indent=2)
    log(f"compiled {name} in {entry['compile_s']:.0f}s alloc={entry['alloc_gib']}")
