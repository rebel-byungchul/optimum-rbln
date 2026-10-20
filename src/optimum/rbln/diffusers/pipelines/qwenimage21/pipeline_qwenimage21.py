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

import math
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils.torch_utils import randn_tensor

from ...configurations import RBLNQwenImage21PipelineConfig
from ...configurations.pipelines.configuration_qwenimage21 import QWENIMAGE21_CONDITION_RESOLUTION, qwenimage21_size
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNQwenImage21Pipeline(RBLNDiffusionMixin, QwenImage21Pipeline):
    """
    RBLN-accelerated Qwen-Image-2.1 pipeline for text-to-image generation and image editing.

    The Qwen3-VL prompt encoder (with its vision tower), the transformer and the VAE run on RBLN NPUs, compiled for
    one profile of image sizes (see [`RBLNQwenImage21PipelineConfig`]). Condition images (`image`) are center-cropped
    to the closest compiled aspect ratio and resized to its 1024-class size, which is what the original pipeline would
    resize them to; the generated image takes one of the compiled sizes.

    Example:
        ```python
        from optimum.rbln import RBLNQwenImage21Pipeline

        pipe = RBLNQwenImage21Pipeline.from_pretrained(
            "Qwen/Qwen-Image-2.1-Turbo", export=True, model_save_dir="qwen-image-2.1-rbln",
            rbln_config={
                "profile": "1024",
                "transformer": {"device": [0, 1]},
                "text_encoder": {"device": 0},
                "vae": {"device": [1, 0]},
            },
        )
        image = pipe("A red fox in the snow", width=1024, height=1024).images[0]
        ```
    """

    original_class = QwenImage21Pipeline
    _rbln_config_class = RBLNQwenImage21PipelineConfig
    _submodules = ["transformer", "text_encoder", "vae"]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "RBLNQwenImage21Pipeline":
        # Compiled from float32 weights: the NPU computes a float32 graph in dlfloat16 (9 mantissa bits, against 7 for
        # bfloat16) and keeps the weights in 2 bytes, so this is more accurate at the same memory and, for the
        # transformer, faster (2.75 against 3.15 s for 1024x1024 text-to-image). The VAE casts its own graphs.
        kwargs.setdefault("torch_dtype", torch.float32)
        return super().from_pretrained(model_id, **kwargs)

    def _fit_condition_image(self, image: PIL.Image.Image | np.ndarray) -> PIL.Image.Image:
        """Center-crops a condition image to the closest aspect ratio the VAE encoder was compiled for and resizes it
        to that size (the size the original pipeline gives an image of that aspect ratio)."""
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        ratio = math.log(image.width / image.height)
        height, width = min(
            self.vae.rbln_config.encoder_sample_sizes, key=lambda s: abs(math.log(s[1] / s[0]) - ratio)
        )
        return PIL.ImageOps.fit(image, (width, height), method=PIL.Image.Resampling.LANCZOS)

    def handle_additional_kwargs(self, **kwargs):
        output_resolution = kwargs.get("output_resolution", QWENIMAGE21_CONDITION_RESOLUTION)
        if output_resolution != QWENIMAGE21_CONDITION_RESOLUTION:
            raise NotImplementedError(
                f"Condition images are compiled at the {QWENIMAGE21_CONDITION_RESOLUTION} class only "
                f"(`output_resolution={QWENIMAGE21_CONDITION_RESOLUTION}`); pass `height`/`width` for a larger output."
            )
        image = kwargs.get("image")
        if image is not None:
            image = [self._fit_condition_image(img) for img in (image if isinstance(image, list) else [image])]
            kwargs["image"] = image
        # The size the original pipeline generates at, which the VAE must have been compiled for.
        height, width = kwargs.get("height"), kwargs.get("width")
        if height is None or width is None:
            default = (
                qwenimage21_size(output_resolution, image[-1].width, image[-1].height)
                if image is not None
                else (output_resolution, output_resolution)
            )
            height, width = height or default[0], width or default[1]
        size = (height // 32 * 32, width // 32 * 32)
        if size not in self.vae.rbln_config.sample_sizes:
            raise ValueError(
                f"The pipeline was not compiled for {size[0]}x{size[1]} images ({self.vae.rbln_config.sample_sizes}); "
                "pass `height`/`width` of one of them."
            )
        kwargs["height"], kwargs["width"] = size
        return kwargs

    # The prompt embeddings stay float32 (the original takes the dtype of everything after the prompt encoder from
    # them), while the initial noise is drawn in bfloat16 like the original in bfloat16 (the model card's dtype), so a
    # seed gives the same image as there.
    def prepare_latents(
        self, images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        if latents is None:
            h, w = 2 * (int(height) // (self.vae_scale_factor * 2)), 2 * (int(width) // (self.vae_scale_factor * 2))
            noise = randn_tensor(
                (batch_size, 1, num_channels_latents, h, w), generator=generator, device=device, dtype=torch.bfloat16
            )
            latents = self._pack_latents(noise, batch_size, num_channels_latents, h, w)
        return super().prepare_latents(
            images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=latents
        )

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        """The original `_get_qwen_prompt_embeds`, one prompt at a time: the compiled encoder returns the last decoder
        layer's output before the final norm, which the original gets with a hook on that norm. Encoding the prompts of
        a batch one by one gives each the positions it has in the original's left-padded batch."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [" " if not p else p for p in prompt]  # Qwen has no bos token
        processor_kwargs = {}
        if image is None:
            template = self.prompt_template_t2i
        else:
            slots = "".join(
                f"{' ' if i > 1 else ''}<image{i}><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(1, len(image) + 1)
            )
            template = self.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", slots)
            processor_kwargs["images"] = [self._vision_image(img) for img in image]

        embeds, image_pad_masks = [], []
        for text in prompt:
            inputs = self.processor(text=[template.format(text)], return_tensors="pt", **processor_kwargs)
            encoder_inputs = {
                key: getattr(inputs, key)
                for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
                if getattr(inputs, key, None) is not None
            }
            hidden = self.text_encoder(**encoder_inputs, return_dict=True).logits[0]
            embeds.append(hidden[self._drop_idx :])
            image_pad_masks.append((inputs.input_ids[0] == self._img_token_id)[self._drop_idx :])

        max_len = max(e.shape[0] for e in embeds)
        pad = lambda t, value=0: torch.cat([t, t.new_full((max_len - t.shape[0], *t.shape[1:]), value)])  # noqa: E731
        prompt_embeds = torch.stack([pad(e) for e in embeds])
        attention_mask = torch.stack([pad(torch.ones(e.shape[0], dtype=torch.long)) for e in embeds])
        image_pad_mask = torch.stack([pad(m) for m in image_pad_masks])
        return prompt_embeds, attention_mask, image_pad_mask

    @staticmethod
    def _vision_image(image: PIL.Image.Image) -> PIL.Image.Image:
        # As the original: the vision encoder reads RGBA condition images composited over white.
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        if image.mode != "RGBA":
            return image
        white = PIL.Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        return white


__all__ = [
    "RBLNQwenImage21Pipeline",
]
