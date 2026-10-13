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
    def _program(self, H, W):
        from . import vae as V

        return V.DECODER_PROGRAM if self._n_tiles(H, W) > 1 else V.DECODER_PROGRAM_COARSE

    @staticmethod
    def _n_tiles(H, W):
        """Row tiles for the high-resolution decoder tail: 1 up to the 1024² class, 3 above (2048 class: ≤ 14 GiB → ~5 GiB)."""
        return 1 if H * W <= 64 * 64 else 3

    def _dec_runtimes(self, H, W):
        """name → runtime (or _OnDemand) of the fine decoder program (vae.DECODER_PROGRAM) at latent size H×W."""
        import rebel

        from . import vae as V

        key = (H, W)
        rts = self._dec.get(key)
        if rts is not None:
            return rts
        n = self._n_tiles(H, W)
        prog = self._program(H, W)
        tag = f"dec9-{H}x{W}-t{n}x{V.tile_plan(8 * H, n)[0]}" if n > 1 else f"dec4-{H}x{W}"
        d = os.path.join(self.cache_dir, f"vae-{tag}")
        names = [nm for nm, _, _, _ in prog]
        files = [os.path.join(d, f"{nm}.rbln") for nm in names]
        with _LOCK:
            if not all(os.path.exists(f) for f in files):
                os.makedirs(d, exist_ok=True)
                full = V.QwenImage21VAEDecoder2D(self._model()).eval()
                gs = {nm: F(full).eval() for nm, F, _, _ in prog}
                # shapes from a 2×2 latent (every op keeps its spatial scale), scaled to H×W; tiled steps use the tile height
                with torch.no_grad():
                    env = V.run_program(prog, gs, {"z": torch.zeros(1, 64, 2, 2, dtype=torch.bfloat16)})
                scale = {k: v.shape[2] // 2 for k, v in env.items()}
                T, _ = V.tile_plan(8 * H, n)
                t0 = time.perf_counter()
                for (nm, _, ins, _), f in zip(prog, files, strict=True):
                    if os.path.exists(f):
                        continue
                    tiled = nm in V.TILED_STEPS and n > 1
                    info = [
                        (
                            k,
                            [1, env[k].shape[1], T * scale[k] // 8 if tiled else H * scale[k], W * scale[k]],
                            "bfloat16",
                        )
                        for k in ins
                    ]
                    rebel.compile_from_torch(gs[nm], input_info=info, npu="RBLN-CR13").save(f)
                self.log(f"[vae] compiled decoder {H}x{W} (latent, {n} row tiles) in {time.perf_counter() - t0:.0f}s")
        self._dec.make_room()
        rts = dict(zip(names, self._load(files), strict=True))
        self._dec.put(key, rts)
        return rts

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
        """Precompile decoder/encoder graphs for an image size in pixels (rounded down to a multiple of 32, same as the original pipeline)."""
        h, w = height // 32 * 32, width // 32 * 32
        self._dec_runtimes(h // 16, w // 16)
        self._enc_runtime(h, w)

    def decode(self, z, return_dict=False):
        """z [1, 64, 1, H, W] (denormalized) → image [1, 4, 1, 16H, 16W]"""
        from contextlib import contextmanager

        from . import vae as V

        t0 = time.perf_counter()
        z4 = z[:, :, 0].to(torch.bfloat16).contiguous()
        H, W = z4.shape[-2:]
        rts = self._dec_runtimes(H, W)
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

        graphs = {nm: (lambda *a, nm=nm: live[nm].run(*a)) for nm in rts}
        env = V.run_program(self._program(H, W), graphs, {"z": z4}, n_tiles=self._n_tiles(H, W), use=use)
        out = _first(env["out"])
        self.timings.setdefault("vae_decode", []).append(time.perf_counter() - t0)
        return (out.unsqueeze(2).float(),)

    # -------------------------------------------------------------- encode (condition image)
    def _enc_runtime(self, H, W):
        import rebel

        from . import vae as V

        key = (H, W)
        rts = self._enc.get(key)
        if rts is not None:
            return rts
        d = os.path.join(self.cache_dir, f"vae-enc-{H}x{W}")
        files = [os.path.join(d, f"e{i}.rbln") for i in range(1, 5)]
        with _LOCK:
            if not all(os.path.exists(f) for f in files):
                os.makedirs(d, exist_ok=True)
                t0 = time.perf_counter()
                enc = V.QwenImage21VAEEncoder2D(self._model()).eval()
                e1, e2, e3, e4 = (
                    M(enc).eval() for M in (V.VAEEncChainE1, V.VAEEncChainE2, V.VAEEncChainE3, V.VAEEncChainE4)
                )
                x = torch.zeros(1, 4, H, W, dtype=torch.bfloat16)
                with torch.no_grad():
                    x1 = e1(x)
                    y, h = e2(x1)
                    x3 = e3(y, h, x1)
                bf = "bfloat16"
                infos = [
                    [("x", [1, 4, H, W], bf)],
                    [("x", list(x1.shape), bf)],
                    [("y", list(y.shape), bf), ("h", list(h.shape), bf), ("x", list(x1.shape), bf)],
                    [("x", list(x3.shape), bf)],
                ]
                for g, info, f in zip((e1, e2, e3, e4), infos, files, strict=False):
                    if not os.path.exists(f):
                        rebel.compile_from_torch(g, input_info=info, npu="RBLN-CR13").save(f)
                self.log(f"[vae] compiled encoder {H}x{W} in {time.perf_counter() - t0:.0f}s")
        self._enc.make_room()
        rts = self._load(files, self.enc_device)
        self._enc.put(key, rts)
        return rts

    def encode(self, x, return_dict=True):
        """x [1, 4, 1, H, W] ([-1,1]) → mimics the posterior (only mode() is used)"""
        t0 = time.perf_counter()
        x4 = x[:, :, 0].to(torch.bfloat16).contiguous()
        r1, r2, r3, r4 = self._enc_runtime(*x4.shape[-2:])
        a = _first(r1.run(x4))
        y, h = r2.run(a)
        b = _first(r3.run(y, h, a))
        z = _first(r4.run(b)).float().unsqueeze(2)
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
        self._rt[NB] = rebel.Runtime(f, tensor_type="pt", device=self.device)
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
            m, d = self._runtime(NB).run(pp, I, Wt, P, torch.tensor([nv], dtype=torch.int32))
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
        self.loaded = OrderedDict()
        self.pinned = set()
        self.load_s = 0.0
        for n in pinned:
            self.pin(n)

    def pin(self, name):
        """The runtime that first allocates the weights and static cache must not be unloaded (the cache contents may be lost)."""
        self.pinned.add(name)
        return self[name]

    def __getitem__(self, name):
        import rebel

        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]
        while True:
            try:
                t0 = time.perf_counter()
                rt = rebel.Runtime(self.paths[name], tensor_type="pt", device=self.device)
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
