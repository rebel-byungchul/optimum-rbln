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

"""VAE decoder/encoder and vision encoder runtimes — each size is compiled on first use, saved to disk, and reused.

For the VAE, masked buckets fail to compile on CR13 (10-13), so each resolution is compiled at its exact size (4-graph chain, about
30-60 s per size). The vision encoder uses patch buckets (trailing padding + key mask), so a few buckets cover all sizes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import torch


_LOCK = threading.Lock()


def _first(x):
    return x[0] if isinstance(x, list | tuple) else x


VERIFY_STATS = {"mismatch": 0}  # how many repeats disagreed (all components except the DiT, which counts its own)


def run_verified(fn, max_runs=7):
    """Call `fn` until two results agree bit for bit and return that result.

    Why (measured 10-14 on CR13): large graphs (2048-class DiT layer groups, 2048-class VAE graphs) now and then return a
    different result for the same input, sometimes with values far out of range; single ops and the 1024-class graphs
    never did. The rate varies over time, so it is checked on every call."""

    def as_list(o):
        return [t.clone() for t in o] if isinstance(o, list | tuple) else [o.clone()]

    seen = [as_list(fn())]
    while True:
        o = as_list(fn())
        for prev in seen:
            if all(torch.equal(a, b) for a, b in zip(o, prev, strict=True)):
                return o if len(o) > 1 else o[0]
        seen.append(o)
        VERIFY_STATS["mismatch"] += 1
        if len(seen) >= max_runs:
            raise RuntimeError(f"graph output does not repeat over {max_runs} runs")


def chiplet0_free(device):
    from rebel.device_info import get_device_memory_info

    return get_device_memory_info(device).chiplets[0].free / 2**30


def load_runtime(path, device, pressure=()):
    """rebel.Runtime with room made on chiplet 0 first (the binding budget on CR13; a load that fails half way does not
    give its memory back). `pressure`: callables that free other components' memory, cheapest first. The chiplet-0
    footprint is measured on the first load and kept next to the file (compile records underestimate graphs that share
    weights: the text encoder's multimodal graph records 0.3 GiB and takes 3.7)."""
    import gc

    import rebel

    side = path + ".chiplet0.json"
    need = json.load(open(side))["gib"] if os.path.exists(side) else None
    if need is not None:
        for free in [None, *pressure]:
            if free is not None:
                free()
                gc.collect()
            if chiplet0_free(device) >= need + 0.3:
                break
    before = chiplet0_free(device)
    rt = rebel.Runtime(path, tensor_type="pt", device=device)
    if need is None:
        try:
            with open(side, "w") as f:
                json.dump({"gib": round(before - chiplet0_free(device), 3)}, f)
        except OSError:
            pass
    return rt


class _OnDemand:
    """A graph that is loaded for one call and freed right after (used when the whole chain does not fit next to the DiT)."""

    def __init__(self, path, device):
        self.path, self.device = path, device

    def open(self):
        import rebel

        return rebel.Runtime(self.path, tensor_type="pt", device=self.device)

    def run(self, *args):
        rt = self.open()
        try:
            return rt.run(*args)
        finally:
            del rt


class _LRU:
    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def make_room(self):
        """Free entries so that one more fits (device memory is released before the next chain is loaded)."""
        import gc

        while self.d and len(self.d) >= self.n:
            self.d.popitem(last=False)
        gc.collect()

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.n:
            self.d.popitem(last=False)


class VAERuntime:
    """Drop-in replacement for diffusers AutoencoderKLQwenImage21. Runs decode/encode as NPU chains."""

    def __init__(self, snapshot, cache_dir, device=0, keep=1, log=print, enc_device=None):
        from diffusers import AutoencoderKLQwenImage21

        self.snapshot, self.cache_dir, self.device, self.log = snapshot, cache_dir, device, log
        self.enc_device = device if enc_device is None else enc_device  # condition-image encoder (runs before the DiT)
        os.makedirs(cache_dir, exist_ok=True)
        cfg = json.load(open(os.path.join(snapshot, "vae", "config.json")))
        from diffusers.configuration_utils import FrozenDict

        self.config = FrozenDict(cfg)
        self.dtype = torch.bfloat16
        self._vae = None  # CPU weights are loaded only when compiling
        self._dec = _LRU(keep)
        self._enc = _LRU(keep)
        self.timings = {}
        self._cls = AutoencoderKLQwenImage21

    def free(self):
        """Drop the cached decoder/encoder runtimes (called when the DiT needs the device memory)."""
        self._dec, self._enc = _LRU(self._dec.n), _LRU(self._enc.n)

    def _model(self):
        if self._vae is None:
            self._vae = self._cls.from_pretrained(self.snapshot, subfolder="vae", torch_dtype=torch.bfloat16).eval()
        return self._vae

    # -------------------------------------------------------------- decode
    # ---------------------------------------------------------------- chain compile (with fallbacks)
    def _candidates(self, kind, H, W):
        """(tag, program, n_tiles, tiling kwargs) to try in order. A size whose graphs do not compile in one piece is
        tiled (row tiles are exact); shape-specific compile errors (e.g. the encoder at 1152x896, 10-14) are skipped."""
        from . import vae as V

        if kind == "dec":
            first = [("dec4", V.DECODER_PROGRAM_COARSE, 1)] if H * W <= 64 * 64 else []
            return first + [("dec12", V.DECODER_PROGRAM, n) for n in (4, 3, 6, 2, 8)], {}
        enc_kw = {"tiled": V.ENC_TILED_STEPS, "halo": V.ENC_TILE_HALO, "base": "x", "unit": V.ENC_TILE_UNIT}
        return [("enc5", V.ENCODER_PROGRAM, n) for n in (1, 2, 3, 4, 6)], enc_kw

    def _chain(self, kind, H, W):
        """Compile (once, cached on disk) and return (dir, program, n_tiles, tiling kwargs) for a chain at size H×W
        (latent size for the decoder, pixels for the encoder)."""
        import shutil

        import rebel

        from . import vae as V

        cands, kw = self._candidates(kind, H, W)
        tiled = kw.get("tiled", V.TILED_STEPS)
        halo, unit = kw.get("halo", V.TILE_HALO), kw.get("unit", 1)
        rows = H // unit
        for tag, prog, n in cands:
            T = V.tile_plan(rows, n, halo)[0]
            d = os.path.join(self.cache_dir, f"vae-{tag}-{H}x{W}" + (f"-t{n}x{T}" if n > 1 else ""))
            files = [os.path.join(d, f"{nm}.rbln") for nm, _, _, _ in prog]
            if os.path.exists(os.path.join(d, "FAILED")):
                continue
            if all(os.path.exists(f) for f in files):
                return d, prog, n, kw
            with _LOCK:
                os.makedirs(d, exist_ok=True)
                model = (
                    V.QwenImage21VAEDecoder2D(self._model())
                    if kind == "dec"
                    else V.QwenImage21VAEEncoder2D(self._model())
                )
                gs = {nm: F(model.eval()).eval() for nm, F, _, _ in prog}
                # shapes from a small probe (two units high and wide; every op keeps its spatial scale), scaled up
                probe = 2 * unit
                start = (
                    {"z": torch.zeros(1, 64, probe, probe, dtype=torch.bfloat16)}
                    if kind == "dec"
                    else {"x": torch.zeros(1, 4, probe, probe, dtype=torch.bfloat16)}
                )
                with torch.no_grad():
                    env = V.run_program(prog, gs, dict(start))
                t0 = time.perf_counter()
                try:
                    for (nm, _, ins, _), f in zip(prog, files, strict=True):
                        if os.path.exists(f):
                            continue
                        t_rows = T if (nm in tiled and n > 1) else rows
                        info = [
                            (
                                k,
                                [1, env[k].shape[1], t_rows * env[k].shape[2] // 2, W * env[k].shape[3] // probe],
                                "bfloat16",
                            )
                            for k in ins
                        ]
                        rebel.compile_from_torch(gs[nm], input_info=info, npu="RBLN-CR13").save(f)
                except Exception as e:  # noqa: BLE001
                    self.log(
                        f"[vae] {kind} {H}x{W} with {n} tile(s) does not compile ({str(e)[:60]}); trying the next layout"
                    )
                    for f in files:
                        if os.path.exists(f):
                            os.remove(f)
                    open(os.path.join(d, "FAILED"), "w").write(str(e)[:500])
                    continue
                self.log(f"[vae] compiled {kind} {H}x{W} ({n} row tile(s)) in {time.perf_counter() - t0:.0f}s")
                shutil.rmtree(os.path.join(d, "__tmp__"), ignore_errors=True)
                return d, prog, n, kw
        raise RuntimeError(f"VAE {kind} at {H}x{W}: no graph layout compiles")

    def _dec_runtimes(self, H, W):
        key = (H, W)
        hit = self._dec.get(key)
        if hit is not None:
            return hit
        d, prog, n, kw = self._chain("dec", H, W)
        names = [nm for nm, _, _, _ in prog]
        self._dec.make_room()
        rts = dict(zip(names, self._load([os.path.join(d, f"{nm}.rbln") for nm in names]), strict=True))
        hit = (rts, prog, n, kw)
        self._dec.put(key, hit)
        return hit

    @staticmethod
    def _chiplet0_gib(files):
        """Chiplet-0 GiB of each compiled graph (cached in chiplet0.json next to them)."""
        import rebel

        meta = os.path.join(os.path.dirname(files[0]), "chiplet0.json")
        sizes = json.load(open(meta)) if os.path.exists(meta) else {}
        for f in files:
            k = os.path.basename(f)
            if k not in sizes:
                by = rebel.RBLNCompiledModel(f).get_alloc_per_chiplet_by_key()
                sizes[k] = sum(v[0][0] for v in by.values()) / 2**30
                with open(meta, "w") as fh:
                    json.dump(sizes, fh, indent=1)
        return [sizes[os.path.basename(f)] for f in files]

    def _load(self, files, device=None):
        """Load a chain resident if it fits in the free chiplet-0 memory (the binding budget on CR13), else load each graph only
        for its own call (2048-class sizes next to the DiT). Deciding up front avoids failed loads, which leak device memory."""
        import gc

        import rebel
        from rebel.device_info import get_device_memory_info

        device = self.device if device is None else device
        gc.collect()
        need = self._chiplet0_gib(files)
        free = get_device_memory_info(device).chiplets[0].free / 2**30 - 0.5  # margin
        if sum(need) <= free:
            return [rebel.Runtime(f, tensor_type="pt", device=device) for f in files]
        self.log(
            f"[vae] {os.path.basename(os.path.dirname(files[0]))}: needs {sum(need):.1f} GiB of chiplet 0, "
            f"{free:.1f} free: loading graphs per call"
        )
        return [_OnDemand(f, device) for f in files]

    def compile_size(self, height, width):
        """Precompile decoder/encoder graphs for an image size in pixels (rounded down to a multiple of 32, as the original)."""
        h, w = height // 32 * 32, width // 32 * 32
        self._chain("dec", h // 16, w // 16)
        self._chain("enc", h, w)

    def _run_chain(self, hit, start):
        from contextlib import contextmanager

        from . import vae as V

        rts, prog, n, kw = hit
        live = {}

        @contextmanager
        def use(name):  # an on-demand graph is loaded once per step (shared by its row tiles) and freed after
            r = rts[name]
            live[name] = r.open() if isinstance(r, _OnDemand) else r
            try:
                yield
            finally:
                if isinstance(r, _OnDemand):
                    del live[name]

        def call(nm, *a):
            return run_verified(lambda: live[nm].run(*a))  # (see run_verified)

        graphs = {nm: (lambda *a, nm=nm: call(nm, *a)) for nm in rts}
        return V.run_program(prog, graphs, start, n_tiles=n, use=use, **kw)

    def decode(self, z, return_dict=False):
        """z [B, 64, 1, H, W] (denormalized) → image [B, 4, 1, 16H, 16W]"""
        if z.shape[0] > 1:  # graphs are compiled for one image: decode a batch image by image
            return (torch.cat([self.decode(z[i : i + 1])[0] for i in range(z.shape[0])]),)
        t0 = time.perf_counter()
        z4 = z[:, :, 0].to(torch.bfloat16).contiguous()
        env = self._run_chain(self._dec_runtimes(*z4.shape[-2:]), {"z": z4})
        out = _first(env["out"])
        self.timings.setdefault("vae_decode", []).append(time.perf_counter() - t0)
        return (out.unsqueeze(2).float(),)

    # -------------------------------------------------------------- encode (condition image)
    def _enc_runtimes(self, H, W):
        key = (H, W)
        hit = self._enc.get(key)
        if hit is not None:
            return hit
        d, prog, n, kw = self._chain("enc", H, W)
        names = [nm for nm, _, _, _ in prog]
        self._enc.make_room()
        rts = dict(
            zip(names, self._load([os.path.join(d, f"{nm}.rbln") for nm in names], self.enc_device), strict=True)
        )
        hit = (rts, prog, n, kw)
        self._enc.put(key, hit)
        return hit

    def encode(self, x, return_dict=True):
        """x [1, 4, 1, H, W] ([-1,1]) → mimics the posterior (only mode() is used)"""
        t0 = time.perf_counter()
        x4 = x[:, :, 0].to(torch.bfloat16).contiguous()
        env = self._run_chain(self._enc_runtimes(*x4.shape[-2:]), {"x": x4})
        z = _first(env["z"]).float().unsqueeze(2)
        self.timings.setdefault("vae_encode", []).append(time.perf_counter() - t0)
        dist = SimpleNamespace(mode=lambda: z, sample=lambda generator=None: z)
        return SimpleNamespace(latent_dist=dist)


class VisionRuntime:
    """Qwen3-VL vision encoder. Patch-bucket graphs (one image per run)."""

    BUCKETS = (4352, 17408)

    def __init__(self, snapshot, cache_dir, device=0, log=print):
        self.snapshot, self.cache_dir, self.device, self.log = snapshot, cache_dir, device, log
        os.makedirs(cache_dir, exist_ok=True)
        self._visual = None
        self._rt = {}
        self.pressure = []  # callables that free other components' device memory (set by the pipeline)
        self.timings = {}
        vc = json.load(open(os.path.join(snapshot, "text_encoder", "config.json")))["vision_config"]
        m = vc.get("spatial_merge_size", 2)
        # only the values needed for metadata computation (no weights needed)
        self.meta = SimpleNamespace(
            num_grid_per_side=int(vc["num_position_embeddings"] ** 0.5),
            interpolation_mode="bilinear",
            interpolation_align_corners=True,
            spatial_merge_size=m,
            config=SimpleNamespace(spatial_merge_size=m),
        )

    def free(self):
        """Unload the bucket runtimes (loaded again on next use)."""
        import gc

        self._rt = {}
        gc.collect()

    def visual(self):
        """CPU module used for metadata computation (interpolation indices, weights, RoPE positions) and for compilation. Weights are needed only for compilation."""
        if self._visual is None:
            from transformers import Qwen3VLForConditionalGeneration

            te = Qwen3VLForConditionalGeneration.from_pretrained(
                self.snapshot, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
            )
            self._visual = te.model.visual.eval()
            del te
        return self._visual

    def compile_bucket(self, NB):
        return self._runtime(NB)

    def _runtime(self, NB):
        import rebel

        from .vision import VisionStatic

        if NB in self._rt:
            return self._rt[NB]
        f = os.path.join(self.cache_dir, f"vision-{NB}.rbln")
        with _LOCK:
            if not os.path.exists(f):
                t0 = time.perf_counter()
                vis = self.visual()
                mod = VisionStatic(vis).eval()
                cin = vis.patch_embed.proj.weight[0].numel()
                info = [
                    ("patches", [NB, cin], "float32"),
                    ("interp_idx", [NB, 4], "int64"),
                    ("interp_w", [NB, 4], "float32"),
                    ("pos", [2, NB], "int64"),
                    ("n_valid", [1], "int32"),
                ]
                rebel.compile_from_torch(mod, input_info=info, npu="RBLN-CR13").save(f)
                self.log(f"[vision] compiled bucket {NB} in {time.perf_counter() - t0:.0f}s")
        self._rt[NB] = load_runtime(f, self.device, self.pressure)
        return self._rt[NB]

    def encode(self, pixel_values, image_grid_thw):
        """processor output (multiple images concatenated) → merged [Σn/4, 4096], deepstack [3, Σn/4, 4096]"""
        from .vision import vision_metadata

        t0 = time.perf_counter()
        merged, ds, o = [], [], 0
        for g in image_grid_thw:
            n = int(g.prod())
            NB = next(b for b in self.BUCKETS if b >= n)
            I, Wt, P, nv = vision_metadata(self.meta, g[None], NB)
            pp = torch.zeros(NB, pixel_values.shape[1])
            pp[:n] = pixel_values[o : o + n].float()
            rt = self._runtime(NB)
            args = (pp, I, Wt, P, torch.tensor([nv], dtype=torch.int32))
            m, d = run_verified(lambda rt=rt, args=args: rt.run(*args))
            merged.append(m[: n // 4])
            ds.append(d[:, : n // 4])
            o += n
        self.timings.setdefault("vision", []).append(time.perf_counter() - t0)
        return torch.cat(merged, 0), torch.cat(ds, 1)


class LazyRuntimes:
    """Graph name → runtime. Loaded on first use; if device memory runs out (SYS_EBUSY), the least recently used ones are unloaded and loading is retried.
    Graphs in the same weight group share weights and only scratch (Intermediate) is allocated per runtime, so keeping only the graphs
    currently in use loaded fits on a single device."""

    def __init__(self, paths: dict, device: int, pinned=()):
        self.paths, self.device = paths, device
        self.pressure = []  # callables that free other components' device memory (set by the pipeline)
        self.loaded = OrderedDict()
        self.pinned = set()
        self.load_s = 0.0
        for n in pinned:
            self.pin(n)

    def free(self):
        """Unload every runtime (including pinned ones); they are loaded again on their next use."""
        import gc

        self.loaded.clear()
        gc.collect()

    def pin(self, name):
        """The runtime that first allocates the weights and static cache must not be unloaded (the cache contents may be lost)."""
        self.pinned.add(name)
        return self[name]

    def __getitem__(self, name):
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]
        for p in self.pinned:  # the pinned graph's file holds the shared weights: it must be loaded before the others
            if p != name and p not in self.loaded:
                self[p]
        while True:
            try:
                t0 = time.perf_counter()
                rt = load_runtime(self.paths[name], self.device, self.pressure)
                self.load_s += time.perf_counter() - t0
                break
            except RuntimeError as e:
                if "EBUSY" not in str(e) and "device memory" not in str(e):
                    raise
                victims = [k for k in self.loaded if k not in self.pinned]
                if not victims:
                    raise
                del self.loaded[victims[0]]
                import gc

                gc.collect()
        self.loaded[name] = rt
        return rt
