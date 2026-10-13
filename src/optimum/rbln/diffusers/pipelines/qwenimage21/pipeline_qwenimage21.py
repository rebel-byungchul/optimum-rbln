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

"""Qwen-Image-2.1 (Turbo) pipeline on RBLN NPUs.

Every model computation runs on the NPU: the Qwen3-VL text encoder (vision tower + language model), the DiT and the
VAE encoder/decoder. The host only tokenizes, preprocesses images, builds integer metadata (positions, cache slots,
masks) and runs the scheduler update. All features of the original pipeline are supported: text-to-image at any
resolution up to the 2048 class, transparent (RGBA) output, single-image editing, multi-reference editing (up to 10
images), local edits through annotated images, true CFG with a negative prompt and `use_kv_cache` on/off.

Design notes (details in the module docstrings):
- DiT (`dit_kv.py`): prefix (text + condition images) K/V live in device-resident caches written once per request;
  each denoising step only recomputes the target tokens, exactly like the original `kv_cache_mode` extract/cached.
  Attention uses `rbln_custom_ops.paged_(flash_)causal_attn_prefill`, causal for text and bidirectional for image
  blocks, which reproduces the original block-causal mask.
- Text encoder (`te_kv.py`): the Qwen3-VL language model runs chunk by chunk over a paged KV cache; image features and
  deepstack features are injected inside the graph; mRoPE tables are gathered inside the graph.
- VAE (`vae.py`): the Wan-style 3D VAE is rewritten in 2D for single frames and split into a few graphs; each image
  size is compiled the first time it is used and cached on disk.
- Precision: the graphs are compiled from fp32 modules so the NPU computes in dlfloat16. SiLU/GELU-tanh use exp-based
  formulas and the timestep sinusoid uses an exact table lookup, because the default kernels lose accuracy on CR13.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any

import torch


try:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
except ImportError as e:  # diffusers < 0.41 has no Qwen-Image 2.1
    raise ImportError(
        "RBLNQwenImage21Pipeline needs a diffusers version that ships `QwenImage21Pipeline` "
        "(diffusers>=0.41, currently the main branch) and transformers>=5.17."
    ) from e
from PIL import Image as PILImage
from transformers import AutoConfig, Qwen3VLProcessor

from .runtime_aux import VAERuntime, VisionRuntime
from .runtime_dit import UNIFIED, RBLNDiTKV, compile_unified
from .runtime_te import RBLNTextEncoderKV, compile_te_bank


_CONFIG_FILES = [
    "model_index.json",
    "scheduler",
    "processor",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae",
]


# Chiplet-0 memory (GiB, measured on CR13) of the parts that share devices with the DiT: text encoder (4.8 + 0.3), vision
# encoder 2.4 and VAE encoder 3.2 on the aux device; one VAE decoder graph (largest at the 2048 class: 10.0) on the VAE device.
# CR13 DRAM is split into 4 chiplets of 35 GiB and chiplet 0 receives most scratch memory, so it is the binding budget.
AUX_CHIPLET0_GIB = 10.7
VAE_CHIPLET0_GIB = 10.0


class _RopeIndexer:
    """`Qwen3VLModel.get_rope_index` without weights (built on the meta device)."""

    def __init__(self, te_config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        with torch.device("meta"):
            self.m = Qwen3VLModel._from_config(te_config)

    def __call__(self, **kw):
        return self.m.get_rope_index(**kw)


class RBLNQwenImage21Pipeline(QwenImage21Pipeline):
    """RBLN-accelerated Qwen-Image 2.1 pipeline.

    Example:
        ```python
        from optimum.rbln import RBLNQwenImage21Pipeline

        # compile once (takes ~1-2 hours: DiT/text-encoder graph banks)
        pipe = RBLNQwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", export=True,
                                                       model_save_dir="qwen-image-2.1-turbo-rbln")
        # later
        pipe = RBLNQwenImage21Pipeline.from_pretrained("qwen-image-2.1-turbo-rbln")
        image = pipe(prompt="...", width=1696, height=2528, generator=torch.Generator("cpu").manual_seed(42)).images[0]
        ```
    """

    def __init__(self, scheduler, processor, te, vision, transformer, vae, te_config, sample_sigmas=None):
        object.__setattr__(self, "_rbln_parts", {"transformer": transformer, "vae": vae})
        super().__init__(
            scheduler=scheduler,
            vae=None,
            text_encoder=None,
            processor=processor,
            transformer=None,
            sample_sigmas=sample_sigmas,
        )
        self.te, self.vision = te, vision
        self.rope_indexer = _RopeIndexer(te_config)
        self.image_token_id = te_config.image_token_id

    def __getattribute__(self, name):
        if name in ("transformer", "vae"):
            parts = object.__getattribute__(self, "_rbln_parts")
            if parts.get(name) is not None:
                return parts[name]
        return object.__getattribute__(self, name)

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def maybe_free_model_hooks(self):  # components are NPU runtimes, there is no CPU offload hook
        pass

    # Same inputs/outputs as the original `_get_qwen_prompt_embeds`; only the text encoder call runs on the NPU.
    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if len(prompt) != 1:
            raise NotImplementedError("RBLNQwenImage21Pipeline supports batch size 1.")
        prompt = [" " if not p else p for p in prompt]
        is_t2i = image is None
        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
            kw = {}
        else:
            replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(2, len(image) + 1):
                replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
            tmpl = self.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
            prompts = [tmpl.format(t) for t in prompt]
            imgs = []
            for img in image:
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(img)
                if img.mode == "RGBA":  # as the original: only the vision-encoder copy is flattened on white
                    white = PILImage.new("RGB", img.size, (255, 255, 255))
                    white.paste(img, mask=img.getchannel("A"))
                    img = white
                imgs.append(img)
            kw = {"images": imgs}
        mi = self.processor(text=prompts, padding=True, padding_side="left", return_tensors="pt", **kw)
        ids = mi.input_ids[0]
        L = ids.shape[0]
        if is_t2i:
            hidden = self.te.encode(ids, torch.arange(L)[None].repeat(3, 1))
        else:
            merged, ds = self.vision.encode(mi.pixel_values, mi.image_grid_thw)
            pos, _ = self.rope_indexer(
                input_ids=mi.input_ids,
                mm_token_type_ids=mi.mm_token_type_ids,
                image_grid_thw=mi.image_grid_thw,
                attention_mask=mi.attention_mask,
            )
            hidden = self.te.encode(
                ids, pos[:, 0].long(), vis_rows=merged, vis_mask=ids == self.image_token_id, deepstack=ds
            )
        hidden = hidden.to(torch.bfloat16)
        prompt_embeds = hidden[self._drop_idx :][None]
        image_pad_mask = (ids == self._img_token_id)[self._drop_idx :][None]
        return prompt_embeds, None, image_pad_mask

    # ------------------------------------------------------------------------------------ load / compile / save
    @staticmethod
    def _resolve(model_id: str) -> str:
        if os.path.isdir(model_id):
            return model_id
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = False,
        rbln_config: dict[str, Any] | None = None,
        model_save_dir: str | None = None,
        **kwargs,
    ):
        """export=True: compile `model_id` (HF id or local snapshot) into `model_save_dir`, then load it.
        export=False: `model_id` is a directory produced by an earlier export / `save_pretrained`.

        rbln_config keys:
          `dit_device`: device or list of devices for the DiT (default [0, 1]). The 2048-class (eager) graphs need two
            CR13 devices; their 8-layer groups are spread over the list by chiplet-0 memory.
          `aux_device`: text encoder, vision encoder and VAE encoder (default 0).
          `vae_device`: VAE decoder (default: the last DiT device; graphs are loaded per call when memory is short).
          `vae_sizes`: list of [height, width] to precompile (other sizes compile on first use)."""
        rbln_config = dict(rbln_config or {})
        if export:
            save_dir = model_save_dir or os.path.basename(model_id.rstrip("/")) + "-rbln"
            cls.compile(cls._resolve(model_id), save_dir, rbln_config)
            model_id = save_dir
        cfg_path = os.path.join(model_id, "rbln_config.json")
        saved = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        saved.update(rbln_config)
        return cls._load(model_id, saved)

    @classmethod
    def _load(cls, d: str, rbln_config: dict[str, Any]):
        model_index = json.load(open(os.path.join(d, "model_index.json")))
        tcfg = json.load(open(os.path.join(d, "transformer", "config.json")))
        te_config = AutoConfig.from_pretrained(os.path.join(d, "text_encoder"))
        dd, ad = rbln_config.get("dit_device", [0, 1]), rbln_config.get("aux_device", 0)
        vd = rbln_config.get("vae_device", dd[-1] if isinstance(dd, list | tuple) else dd)
        reserve = {ad: AUX_CHIPLET0_GIB}  # chiplet-0 GiB of the other components → DiT layer-group placement
        reserve[vd] = reserve.get(vd, 0.0) + VAE_CHIPLET0_GIB
        pipe = cls(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(d, subfolder="scheduler"),
            processor=Qwen3VLProcessor.from_pretrained(d, subfolder="processor"),
            te=RBLNTextEncoderKV(os.path.join(d, "te-kv-bank"), device=ad),
            vision=VisionRuntime(d, os.path.join(d, "vision"), device=ad),
            transformer=RBLNDiTKV(os.path.join(d, "dit-unified"), tcfg, device=dd, reserve=reserve),
            vae=VAERuntime(d, os.path.join(d, "vae"), device=vd, enc_device=ad),
            te_config=te_config,
            sample_sigmas=model_index.get("sample_sigmas"),
        )
        pipe.transformer.on_oom.append(pipe.vae.free)  # a DiT cfg switch first drops the cached VAE graphs
        return pipe

    @classmethod
    def compile(cls, snapshot: str, save_dir: str, rbln_config: dict[str, Any] | None = None, log=print):
        """Compile every graph bank of `snapshot` into `save_dir` (resumable: finished graphs are skipped)."""
        from diffusers import QwenImage21Transformer2DModel
        from transformers import Qwen3VLForConditionalGeneration

        rbln_config = dict(rbln_config or {})
        os.makedirs(save_dir, exist_ok=True)
        for f in _CONFIG_FILES:  # configs, scheduler, processor and the small VAE weights (for on-demand VAE sizes)
            src, dst = os.path.join(snapshot, f), os.path.join(save_dir, f)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy(src, dst)
        t0 = time.perf_counter()
        with torch.no_grad():
            # DiT: one model with several executors per cfg (eager/flash) and layer group. Models with different layer
            # ranges share no weights, so `parallel=True` compiles them concurrently in subprocesses (60-100 GB RAM each).
            jobs = [(c, l0, l1) for c, (_, groups) in UNIFIED.items() for l0, l1 in groups]
            if rbln_config.get("parallel", False):
                import subprocess
                import sys

                code = (
                    "import sys, torch; from diffusers import QwenImage21Transformer2DModel as T; "
                    "from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_dit import compile_unified; "
                    "m = T.from_pretrained(sys.argv[1], subfolder='transformer', torch_dtype=torch.float32).eval(); "
                    "compile_unified(m, sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))"
                )
                procs = [
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            code,
                            snapshot,
                            os.path.join(save_dir, "dit-unified"),
                            c,
                            str(l0),
                            str(l1),
                        ]
                    )
                    for c, l0, l1 in jobs
                ]
                if any(p.wait() for p in procs):
                    raise RuntimeError("DiT compile failed")
            else:
                dit = QwenImage21Transformer2DModel.from_pretrained(
                    snapshot, subfolder="transformer", torch_dtype=torch.float32
                ).eval()
                for c, l0, l1 in jobs:
                    compile_unified(dit, os.path.join(save_dir, "dit-unified"), c, l0, l1, log=log)
                del dit
            te = Qwen3VLForConditionalGeneration.from_pretrained(
                snapshot, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
            ).eval()
            compile_te_bank(te.model.language_model, os.path.join(save_dir, "te-kv-bank"), log=log)
            vision = VisionRuntime(save_dir, os.path.join(save_dir, "vision"), log=log)
            vision._visual = te.model.visual
            for nb in vision.BUCKETS:
                vision.compile_bucket(nb)
            del te, vision
            vae = VAERuntime(save_dir, os.path.join(save_dir, "vae"), log=log)
            for h, w in rbln_config.get("vae_sizes", [[1024, 1024]]):
                vae.compile_size(h, w)
        json.dump(rbln_config, open(os.path.join(save_dir, "rbln_config.json"), "w"), indent=2)
        log(f"[compile] done in {time.perf_counter() - t0:.0f}s -> {save_dir}")

    def save_pretrained(self, save_dir: str):
        """Compiled artifacts already live in the directory given to `from_pretrained`; this copies them."""
        src = os.path.dirname(self.transformer.bank_dir)
        if os.path.abspath(src) != os.path.abspath(save_dir):
            shutil.copytree(src, save_dir, dirs_exist_ok=True)

    # ------------------------------------------------------------------------------------ timings
    def all_timings(self):
        out = {}
        for part in (self.te, self.vision, self.transformer, self.vae):
            for k, v in getattr(part, "timings", {}).items():
                out[k] = round(sum(v), 3)
        return out

    def reset_timings(self):
        for part in (self.te, self.vision, self.transformer, self.vae):
            part.timings = {}


__all__ = ["RBLNQwenImage21Pipeline"]
