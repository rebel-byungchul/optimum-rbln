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

Cache configurations (CONFIGS), one resident at a time, picked per request in PREFERENCE order (`RBLNDiTKV._cfg_for`):
  flash: 1024-class targets, long prefixes (many references)    eager: 2048-class targets (single 32256-token block)
  xl:    targets ≤ 16320 tokens next to long prefixes           xxl:   2048-class targets next to long prefixes (runtime_paged)
Each configuration is a few layer-group models ("unified": one compiled model with one executor per (kind, bucket)),
spread over the devices by chiplet-0 memory; prefix chunks that do not fit next to them go through prefix-only models
loaded for the prefix phase (PREFIX_MODELS). The timestep modulation is one small graph (dit_mod.rbln).

Mapping to the original pipeline:
  kv_cache_mode="extract" (first step): write the prefix chunks to the cache and compute the target
  kv_cache_mode="cached"  (later steps): compute only the target (reuse prefix K/V)
  kv_cache_mode=None      (use_kv_cache=False): recompute from the prefix every step (same result)
  cache_context("cond"/"uncond"): the two CFG prefixes go in different cache regions; the target region is shared.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import torch

from .dit_kv import KIND_TAG, CacheLayout, DiTKVShared, DiTKVUnified, DiTModulation, KVConfig, Segment, build_segments
from .ops import split_timestep
from .runtime_aux import chiplet0_free, run_verified


CONFIGS = {
    "flash": KVConfig(part=16384, n_part=5, attn="flash"),
    "eager": KVConfig(part=32256, n_part=1, attn="eager"),
    # long prefixes: the last partition holds the target. Attention across partitions is exact for chunks up to 16320
    # tokens and wrong from 16384 tokens on (whatever the partition size), so the target and image buckets stop at 16320
    # (e.g. 2528x1648 = 16274 tokens).
    "xl": KVConfig(part=16384, n_part=3, attn="flash"),
}
# Unified model (per-kind/bucket executors in one compiled model): cfg → (executor list [(kind, C)], layer groups [(l0, l1)])
# flash serves the 1024 class (targets ≤ 4352 tokens, up to 10 references); eager serves larger targets (up to the 2048
# class, prefix + target < 32768). Only one of them is resident; a request that needs the other one swaps them.
# Device memory budget (CR13): DRAM is split into 4 chiplets of 35 GiB and chiplet 0 receives most of the
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
    # one family per large bucket: a request loads only the executors it uses (one 1-layer model with all four buckets is
    # 2.4 GB, so a 32-layer prefix would read 77 GB per request)
    "eager": [([("txt", 512), ("img", 4352), ("img", B)], 1) for B in (8448, 12288, 14336)],
}
NUM_LAYERS = 32
HIDDEN = 4096
LATENT_C = 64


def cache_names(cfg, layer):
    return f"{cfg}_k{layer}", f"{cfg}_v{layer}"


MOD_FILE = "dit_mod.rbln"  # DiTModulation graph (timestep → modulation), one per bank


def compile_modulation(model, out_dir: str, npu="RBLN-CR13"):
    """The timestep → modulation graph of the bank (skipped if present; written atomically, so concurrent jobs may race)."""
    import rebel

    path = os.path.join(out_dir, MOD_FILE)
    if os.path.exists(path):
        return
    os.makedirs(out_dir, exist_ok=True)
    c = rebel.compile_from_torch(
        DiTModulation(DiTKVShared(model)).eval(),
        input_info=[("t", [3], "float32")],
        example_inputs=[split_timestep(torch.tensor([0.5]))[0]],
        npu=npu,
    )
    tmp = f"{path}.{os.getpid()}.part"
    c.save(tmp)
    os.replace(tmp, path)


class RBLNDiTKV:
    """Runtime that replaces `QwenImage21Transformer2DModel`. Mimics only the attributes the pipeline uses.

    bank_dir contains `{cfg}_L{l0}-{l1}.rbln` (+ .json) files built by compile_unified (one model = several (kind, bucket)
    executors, selected by input shape), the xxl halves (compile_paged) and the modulation graph (compile_modulation).

    `device` may be a list: the layer-group models of a cfg are split into contiguous runs of devices by chiplet-0 memory
    (`_placement`). Scratch memory (Init, Intermediate) is allocated per executor and per layer and never shared."""

    # flash first: it covers the 1024 class (targets ≤ 4352 tokens, up to 10 references) and leaves room for the VAE. eager
    # is loaded only for larger targets. One cfg is resident at a time: a switch unloads the other one (and cached VAE graphs)
    # before loading, since a load that fails half way does not give its device memory back.
    # xxl (per-layer weight graphs + shared attention, runtime_paged.py) is the slowest and serves what the others cannot hold.
    PREFERENCE = ("flash", "eager", "xl", "xxl")

    def __init__(
        self, bank_dir, config: dict, device: int = 0, reserve: dict | None = None, local_cache: str | None = None
    ):
        dirs = [bank_dir] if isinstance(bank_dir, str) else list(bank_dir)
        self.bank_dir = dirs[0]
        self.mod_path = next(
            (os.path.join(d, MOD_FILE) for d in dirs if os.path.exists(os.path.join(d, MOD_FILE))), None
        )
        if self.mod_path is None:
            raise FileNotFoundError(f"{MOD_FILE} missing in {dirs}: compile it with compile_modulation")
        self._mod_rt, self._mods = None, {}  # modulation graph runtime; float(t) → [1, 5D] graph input
        self.models = {}  # cfg -> [(l0, l1, path, executors)]
        self.paged = None  # xxl (runtime_paged.PagedDiT)
        self.prefix_models = {}  # cfg -> [(family, l0, l1, path, executors)] (see PREFIX_MODELS)
        self.local_cache = local_cache  # local directory for copies of the prefix models (see _local)
        self.sizes = {}  # path -> chiplet-0 memory (GiB, from the compile record)
        self.reserve = dict(
            reserve or {}
        )  # device -> chiplet-0 GiB of other components (text encoder, VAE), for _placement
        self.prompt_lens = []  # text lengths of the request's prompts (set by the pipeline; see _cfg_for)
        self._bucket_cache = {}
        for d in dirs:
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    e = json.load(open(os.path.join(d, f)))
                    if e.get("paged"):  # xxl half (runtime_paged.py)
                        self._add_paged_half(d, e)
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
        if self.paged is not None:
            self.paged.devices = self.devices
        self.loaded = {}  # cfg -> [runtime...]
        from diffusers.configuration_utils import FrozenDict

        self.config = FrozenDict(config)
        self.transformer_blocks = [None] * max(m[-1][1] for m in self.models.values())
        self.dtype = torch.bfloat16
        self._ctx = "cond"
        self._state = {}
        self.timings: dict[str, list[float]] = {}
        self.load_s = 0.0
        # cfgs whose model runs are verified by repetition (runtime_aux.run_verified): the 2048-class models. The 1024-class
        # flash models never disagreed, so they run once (add "flash" to check them); xxl verifies its own layer graphs
        self.verify = {"eager", "xl"}
        self.on_oom = []  # callables that free other device memory (e.g. cached VAE graphs) before a cfg switch retries
        # load the cfg that serves the default 1024 class (xxl only on demand: it fills the devices, and the pipeline's
        # memory hooks are not attached yet)
        first = next((c for c in self.PREFERENCE if c in self.models and c != "xxl"), None)
        if first is not None:
            self._runtimes(first)

    @contextmanager
    def cache_context(self, name):
        prev, self._ctx = self._ctx, name
        try:
            yield
        finally:
            self._ctx = prev

    def _runtimes(self, cfg):
        """Layer-group runtimes of a cfg. Only one cfg is resident: the other one is unloaded first. Room on chiplet 0 is made
        up front (a load that fails half way does not give its memory back), cheapest first: `on_oom` hooks."""
        import gc

        import rebel

        if cfg in self.loaded:
            rts = self.loaded[cfg]
            if any(r is None for r in rts):  # xxl graphs unloaded by release_some: load just those
                self._reload_missing(cfg)
            if cfg == "xxl" and self.paged.rts is None:
                self.paged.bind(rts)
            return rts
        t0 = time.perf_counter()
        if self.loaded:  # (self._state already describes the request that triggered this load: keep it)
            self.loaded.clear()
            if self.paged is not None:
                self.paged.rts = None
            gc.collect()
        place = [(p, d) for p, d, _ in self.paged.placement()] if cfg == "xxl" else self._placement(cfg)
        need = {}
        for p, d in place:
            need[d] = need.get(d, 0.0) + self.sizes[p] + 0.3
        for free in [None, *self.on_oom]:
            if free is not None:
                free()
            if all(chiplet0_free(d) >= n for d, n in need.items()):
                break
        rts = [rebel.Runtime(p, tensor_type="pt", device=dev) for p, dev in place]
        self.load_s += time.perf_counter() - t0
        self.loaded[cfg] = rts
        if cfg == "xxl":
            self.paged.bind(rts)
        return rts

    def _local(self, path):
        """Prefix models are read on every request that needs them (a load reads the whole file, mostly Init constants):
        keep a copy on local disk when `local_cache` is set. The copy is keyed by the source path, size and mtime (every
        bank uses the same file names), so a recompiled or different bank never reuses an old copy."""
        import hashlib
        import shutil

        if not self.local_cache:
            return path
        src = os.path.realpath(path)
        st = os.stat(src)
        tag = hashlib.sha1(f"{src}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
        dst = os.path.join(self.local_cache, f"{tag}-{os.path.basename(path)}")
        if not os.path.exists(dst):
            os.makedirs(self.local_cache, exist_ok=True)
            if shutil.disk_usage(self.local_cache).free < os.path.getsize(path) + (50 << 30):
                return path  # shared local disk: keep 50 GB free
            shutil.copyfile(path, dst + ".part")
            os.replace(dst + ".part", dst)
        return dst

    def release_some(self, device, gib):
        """Make about `gib` of chiplet 0 free on `device` by unloading xxl layer graphs (last layers first; they hold no
        caches, so the prefix stays valid) — reloaded by the next call. Other configurations are unloaded whole."""
        import gc

        if "xxl" not in self.loaded:
            return self.release()
        rts, place = self.loaded["xxl"], self.paged.placement()
        freed = 0.0
        for i in reversed(range(len(place))):
            p, d, fe = place[i]
            if freed >= gib:
                break
            if d == device and fe["role"] == "mid" and rts[i] is not None:
                rts[i] = None
                freed += self.sizes[p]
        self.paged.rts = None
        gc.collect()

    def _reload_missing(self, cfg):
        import rebel

        t0 = time.perf_counter()
        rts, place = self.loaded[cfg], self.paged.placement()
        need = {}
        for (p, d, _), r in zip(place, rts, strict=True):
            if r is None:
                need[d] = need.get(d, 0.0) + self.sizes[p] + 0.1
        for free in [None, *self.on_oom]:
            if free is not None:
                free()
            if all(chiplet0_free(d) >= n for d, n in need.items()):
                break
        for i, (p, d, _) in enumerate(place):
            if rts[i] is None:
                rts[i] = rebel.Runtime(p, tensor_type="pt", device=d)
        self.paged.rts = None
        self.load_s += time.perf_counter() - t0
        self.timings.setdefault("dit_reload_part", []).append(time.perf_counter() - t0)

    def release(self):
        """Unload the resident DiT (another component needs the device memory); it is loaded again on its next call."""
        import gc

        self.loaded.clear()
        if self.paged is not None:
            self.paged.rts = None  # (it holds the xxl runtimes too)
        gc.collect()

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
        key = (cfg, kind)
        if key not in self._bucket_cache:
            execs = list(self.models[cfg][0][3]) + [x for pm in self.prefix_models.get(cfg, []) for x in pm[4]]
            self._bucket_cache[key] = sorted({C for k, C in execs if k == kind})
        return self._bucket_cache[key]

    def _pick(self, cfg, kind, n):
        for C in self._buckets(cfg, kind):
            if C >= n:
                return C
        return None

    def _modulation(self, t):
        """[1, 5D] input of the layer graphs for timestep t (modulation | output-norm scale), one graph call per distinct t
        (the prefix uses t = 0, like the original)."""
        import rebel

        key = float(t)
        if key not in self._mods:
            if self._mod_rt is None:
                self._mod_rt = rebel.Runtime(self.mod_path, tensor_type="pt", device=self.devices[0])
            if len(self._mods) > 64:
                self._mods.clear()
            mod, nscale = self._mod_rt.run(split_timestep(torch.tensor([key]))[0])
            self._mods[key] = torch.cat([mod, nscale], dim=-1)
        return self._mods[key]

    def _run(self, cfg, kind, C, x, t, rope, pos, mask):
        l0 = self.load_s
        t_start = time.perf_counter()
        rts = self._runtimes(cfg)
        args = (torch.zeros(KIND_TAG[kind]), self._modulation(t if kind == "tgt" else 0.0), rope, pos)
        bt = torch.arange(CONFIGS[cfg].n_part, dtype=torch.int16)
        out = x
        verify = cfg in self.verify
        for rt in rts:
            out = self._run_group(rt, (out, *args, bt, mask), verify)
        self.timings.setdefault("dit_runtime_load", []).append(self.load_s - l0)
        self.timings.setdefault(f"dit_run_{kind}{C}", []).append(time.perf_counter() - t_start)
        return out

    @staticmethod
    def _run_group(rt, args, verify, multi=False):
        """Run one layer-group model; with `verify`, until two results agree bit for bit (runtime_aux.run_verified: models
        that process 2048-class chunks now and then give a different output for identical inputs)."""
        o = run_verified(lambda: rt.run(*args), key="dit") if verify else rt.run(*args)
        o = list(o) if isinstance(o, list | tuple) else [o]
        return o if multi else o[0]

    def _tick(self, k, t0):
        self.timings.setdefault(k, []).append(time.perf_counter() - t0)

    # ------------------------------------------------------------------ prefix
    def _cfg_for(self, n_tgt, segs):
        """The first cfg in PREFERENCE that has a target bucket and room for the prefix next to the target region. The
        prefix is checked with the longest prompt of the request (`prompt_lens`, e.g. a longer negative prompt), so the
        uncond context fits too."""
        extra = max(self.prompt_lens, default=0) - sum(s.length for s in segs if s.kind == "txt")
        if extra > 0:
            last_txt = max(i for i, s in enumerate(segs) if s.kind == "txt")
            s = segs[last_txt]
            segs = [*segs[:last_txt], Segment("txt", s.start, s.length + extra, s.rope_rows), *segs[last_txt + 1 :]]
        for cfg in self.PREFERENCE:
            if cfg not in self.models:
                continue
            if cfg == "xxl":
                try:
                    self.paged.capacity_check(segs, n_tgt)
                    return cfg
                except ValueError:
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
                self._run(cfg, kind, C, x, 0.0, rope, pos, m)
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
        mod0 = self._modulation(0.0)
        hs = [x for _, _, x, *_ in chunks]
        verify = cfg in self.verify
        for fam, a, _, path, _ in self.prefix_models[cfg]:
            if fam != family:
                continue
            t0 = time.perf_counter()
            rt = rebel.Runtime(self._local(path), tensor_type="pt", device=dev[a])
            self.timings.setdefault("dit_prefix_load", []).append(time.perf_counter() - t0)
            for i, (kind, _, _, rope, pos, m) in enumerate(chunks):
                hs[i] = self._run_group(rt, (hs[i], torch.zeros(KIND_TAG[kind]), mod0, rope, pos, bt, m), verify)
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
            self.prompt_lens = []
            self._state = {"cfg": cfg, "cursor": 0, "all_volatile": False, "ctx": {}}
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
        return (torch.stack(outs).float(),)

    def _tgt_start(self, cfg, C_t, prefix_end):
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
                elif s.kind == "img":
                    lay.place(self._pick(cfg, "img", s.length))
            end = lay.cursor
            self._tgt_start(cfg, C_t, end)
        except (TypeError, ValueError):
            return None
        return end

    @staticmethod
    def _split_inputs(hs, enc_all, enc_mask, layout, N):
        """One sample's transformer inputs → condition latents, target latents, text rows and their validity. Like the
        original, image slot rows of encoder_hidden_states are replaced by latents, so only text rows go into txt_in."""
        n_cond = hs.shape[0] - N
        slot = layout[: enc_all.shape[0]]
        enc = enc_all[~slot]
        txt_valid = torch.ones(enc.shape[0], dtype=torch.bool) if enc_mask is None else enc_mask[~slot].bool()
        return hs[:n_cond], hs[n_cond:], enc, txt_valid

    def _call_one(self, key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode):
        st_all = self._state
        cfg = st_all["cfg"]
        if cfg == "xxl":
            return self._call_one_paged(key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode)
        tgt = segs[-1]
        N = tgt.length
        cond_lat, tgt_lat, enc, txt_valid = self._split_inputs(hs, enc_all, enc_mask, layout, N)
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
        out = self._run(cfg, "tgt", C_t, x, t, rope, (ts_ + torch.arange(C_t, dtype=torch.int32))[None], mask)
        self._tick("dit_target", t0)
        return out[:N]

    # ------------------------------------------------------------------ xxl: weight graphs + shared attention (runtime_paged.py)
    def _add_paged_half(self, d, e):
        from .runtime_paged import PagedDiT

        halves = [*getattr(self, "_paged_halves", []), (e, d)]
        self._paged_halves = halves
        for fe in e["files"]:
            self.sizes[os.path.join(d, fe["name"] + ".rbln")] = fe["chiplet_gib"][0]
        self.paged = PagedDiT(halves, [0], self._run_group)  # devices are set once known (__init__)
        self.models["xxl"] = [(r["l0"], r["l1"], None, []) for r, _ in sorted(halves, key=lambda h: h[0]["l0"])]

    def _call_one_paged(self, key, hs, t, enc_all, enc_mask, layout, segs, kv_cache_mode):
        """_call_one for xxl: each context gets its own block region for its prefix (rewritten from block 0 on every call
        when the regions run out); the target uses the last two blocks of every layer."""
        from .runtime_paged import BlockLayout

        pg = self.paged
        self._runtimes("xxl")  # load (and bind) the xxl graphs if another configuration or nothing is resident
        st_all = self._state
        tgt = segs[-1]
        N = tgt.length
        cond_lat, tgt_lat, enc, txt_valid = self._split_inputs(hs, enc_all, enc_mask, layout, N)
        nbl, part, P = pg.nbl, pg.part, pg.P
        ctxs = st_all["ctx"]
        st = ctxs.get(key)
        if st is None:
            need = pg.capacity_check(segs, N)
            st = {"start": 0, "end": nbl - 2, "volatile": True}
            if not st_all["all_volatile"] and st_all["cursor"] + need <= nbl - 2:
                st = {"start": st_all["cursor"], "end": st_all["cursor"] + need, "volatile": False}
                st_all["cursor"] += need
            else:
                st_all["all_volatile"] = True
            ctxs[key] = st
        if st_all["all_volatile"]:
            st.update(start=0, end=nbl - 2, volatile=True)
        if st.get("valid") is None or kv_cache_mode is None or st["volatile"]:
            t0 = time.perf_counter()
            valid = torch.zeros(nbl, part)
            lay = BlockLayout(st["start"], st["end"], part)
            t_ptr = i_ptr = 0
            zero = torch.zeros(1)
            chunks = []  # (kind, x, rope, place, [(block, offset, rows valid)])
            for kind, C, s, o in pg.chunk_plan(segs):
                if s.kind == "txt":
                    n = min(C, s.length - o)
                    rows = enc[t_ptr + o : t_ptr + o + n]
                    ok = txt_valid[t_ptr + o : t_ptr + o + n].float()
                    if o + n == s.length:
                        t_ptr += s.length
                    rope_rows = s.rope_rows[:, o : o + n]
                else:
                    n = s.length
                    rows, ok = cond_lat[i_ptr : i_ptr + n], torch.ones(n)
                    i_ptr += n
                    rope_rows = s.rope_rows
                if kind != "block":
                    x = torch.zeros(C, rows.shape[-1])
                    x[:n] = rows.float()
                    rope = torch.zeros(3, C, dtype=torch.long)
                    rope[:, :n] = rope_rows
                    pos = lay.place(C)
                    marks = [(pos // part, pos % part, ok)]
                    place = {"blocks": lay.blocks_upto(pos), "s": pos}
                else:
                    sa = lay.place(P)
                    sb = lay.place(P)
                    x, rope, (nA, nB) = self._piece_rows(rows, rope_rows, P)
                    marks = [(sa // part, sa % part, torch.ones(nA)), (sb // part, sb % part, torch.ones(nB))]
                    place = {"before": lay.blocks_upto(sa)[:-1], "sa": sa, "sb": sb}
                chunks.append((kind, x, rope, place, marks))
            # groups of chunks share the layer graph calls; a group's positions become valid only when it runs (a split
            # block's swapped block order must not expose later chunks that share its blocks)
            for grp in pg.pack(chunks):
                for *_, marks in grp:
                    for b, off, ok in marks:
                        valid[b, off : off + len(ok)] = ok
                pg.run_group_of([c[:4] for c in grp], zero, valid)
            last = (lay.cursor - 1) // part if lay.cursor > st["start"] * part else st["start"] - 1
            st["valid"], st["blocks"] = valid, list(range(st["start"], last + 1))
            self._tick("dit_prefix", t0)
        x, rope, (nA, nB) = self._piece_rows(tgt_lat, tgt.rope_rows, P)
        valid = st["valid"].clone()
        valid[nbl - 2] = 0
        valid[nbl - 1] = 0
        valid[nbl - 2, :nA] = 1
        valid[nbl - 1, :nB] = 1
        place = {"before": st["blocks"], "sa": (nbl - 2) * part, "sb": (nbl - 1) * part}
        t0 = time.perf_counter()
        out = pg.run_chunk("block", x, rope, t, place, valid, target=True)
        self._tick("dit_target", t0)
        return torch.cat([out[:nA], out[P : P + nB]])

    @staticmethod
    def _piece_rows(rows, rope_rows, P):
        """A block of N rows → 2P rows: A = first ceil(N/2) rows, B = the rest, each padded to P."""
        N = rows.shape[0]
        nA = (N + 1) // 2
        nB = N - nA
        x = torch.zeros(2 * P, rows.shape[-1])
        x[:nA], x[P : P + nB] = rows[:nA].float(), rows[nA:].float()
        rope = torch.zeros(3, 2 * P, dtype=torch.long)
        rope[:, :nA], rope[:, P : P + nB] = rope_rows[:, :nA], rope_rows[:, nA:]
        return x, rope, (nA, nB)


def compile_unified(
    model,
    out_dir: str,
    cfg_name: str,
    l0: int,
    l1: int,
    npu="RBLN-CR13",
    log=print,
    num_layers=None,
):
    """Layers [l0, l1) of one cfg as a single compiled model with multiple (kind, bucket) executors, plus the cfg's
    PREFIX_MODELS for the same layers, all in one compile context (so they share caches and weights). Models with different
    layer ranges share nothing, so they can be compiled concurrently in separate processes. Also compiles the bank's
    modulation graph (compile_modulation) if it is missing."""
    import rebel
    from rebel.compile_context import CompileContext

    nl = num_layers or NUM_LAYERS
    cfg = CONFIGS[cfg_name]
    execs = UNIFIED[cfg_name][0]
    compile_modulation(model, out_dir, npu)
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
                ("mod", [1, 5 * HIDDEN], "float32"),
                ("rope_idx", [3, C], "int64"),
                ("cache_pos", [1, C], "int32"),
                ("block_tables", [cfg.n_part], "int16"),
                ("mask", [1, cfg.cap], "float32"),
            ]
            ex = [
                torch.zeros(C, xd),
                torch.zeros(KIND_TAG[kind]),
                torch.zeros(1, 5 * HIDDEN),
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
