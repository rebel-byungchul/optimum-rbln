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

from typing import Any

from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3VLModelConfig
from ..models import RBLNAutoencoderKLQwenImage21Config, RBLNQwenImage21Transformer2DModelConfig


# The resolution presets of the Qwen-Image-2.1 model card as (width, height): the 2048-class image sizes, and the
# aspect ratios of the 1024-class sizes and of the condition images.
QWENIMAGE21_PRESETS = [
    (2048, 2048),
    (2400, 1792),
    (1792, 2400),
    (2528, 1696),
    (1696, 2528),
    (2752, 1536),
    (1536, 2752),
]
# Condition images are resized to this many pixels (the pipeline's `output_resolution` squared).
QWENIMAGE21_CONDITION_RESOLUTION = 1024
# Vision patches of a condition image of the 1024 class (16-pixel patches; 1024x1024 has 4096).
QWENIMAGE21_VISION_PATCHES = 4352


def qwenimage21_size(resolution: int, width: int, height: int) -> tuple[int, int]:
    """(height, width) the pipeline gives a `resolution`-squared image of aspect `width:height`."""
    out_width, out_height, _ = calculate_dimensions(resolution * resolution, width / height)
    return out_height, out_width


class RBLNQwenImage21PipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN Qwen-Image-2.1 pipeline (text-to-image and image editing).

    A compiled pipeline serves one `profile` of image sizes:
      - "1024": images of the 1024 class (the presets' aspect ratios at about 1024x1024 pixels), with up to 10
        condition images.
      - "2048": the 2048-class presets of the model card (e.g. 2048x2048, 2528x1696), with condition images.
    Condition images are of the 1024 class (the pipeline's default `output_resolution`), center-cropped to the closest
    aspect ratio of `aspect_ratios`, since the VAE encoder is compiled per size.

    On RBLN-CR13 the pipeline takes about 105 GiB ("1024") or 110 GiB ("2048") of device memory, the first chiplet of a
    device being the binding budget, so it runs on two devices: the transformer's layer groups spread over both, the
    prompt encoder on one, and the VAE decoder on the other with its encoder next to the prompt encoder, e.g.
    `rbln_config={"transformer": {"device": [0, 1]}, "text_encoder": {"device": 0}, "vae": {"device": [1, 0]}}`.
    """

    submodules = ["transformer", "text_encoder", "vae"]

    def __init__(
        self,
        transformer: RBLNQwenImage21Transformer2DModelConfig | None = None,
        text_encoder: RBLNQwen3VLModelConfig | None = None,
        vae: RBLNAutoencoderKLQwenImage21Config | None = None,
        *,
        profile: str | None = None,
        aspect_ratios: list[tuple[int, int]] | None = None,
        image_sizes: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            transformer (RBLNQwenImage21Transformer2DModelConfig | None): Configuration for the transformer.
                Initialized as RBLNQwenImage21Transformer2DModelConfig with `profile` if not provided.
            text_encoder (RBLNQwen3VLModelConfig | None): Configuration for the Qwen3-VL prompt encoder.
                Initialized as RBLNQwen3VLModelConfig if not provided: prompts of up to 16384 tokens (10 condition
                images of the 1024 class), the last layer's output before the final norm, and the precision options.
            vae (RBLNAutoencoderKLQwenImage21Config | None): Configuration for the VAE.
                Initialized as RBLNAutoencoderKLQwenImage21Config with the image and condition-image sizes.
            profile (str | None): "1024" or "2048". Defaults to "1024".
            aspect_ratios (list[tuple[int, int]] | None): `(width, height)` aspect ratios of the condition images
                and of the 1024-class images. Defaults to those of the model card presets.
            image_sizes (list[tuple[int, int]] | None): `(height, width)` of the images to generate. Defaults to the
                profile's: the 2048-class presets, or the aspect ratios at the 1024 class.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `profile` is unknown.
        """
        super().__init__(**kwargs)
        profile = profile or "1024"
        if profile not in ("1024", "2048"):
            raise ValueError(f"profile must be '1024' or '2048', got {profile!r}")
        aspect_ratios = [tuple(r) for r in aspect_ratios or QWENIMAGE21_PRESETS]
        condition_sizes = sorted({qwenimage21_size(QWENIMAGE21_CONDITION_RESOLUTION, w, h) for w, h in aspect_ratios})
        if image_sizes is None:
            image_sizes = [(h, w) for w, h in QWENIMAGE21_PRESETS] if profile == "2048" else condition_sizes

        self.transformer = self.initialize_submodule_config(
            transformer, cls_name="RBLNQwenImage21Transformer2DModelConfig", profile=profile
        )
        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3VLModelConfig",
            max_seq_len=16384,
            attn_impl="flash_attn",
            kvcache_partition_len=4096,
            # (a 1024-token prefill chunk crashes the compiler on RBLN-CR13, rebel-compiler 0.12.1.dev163)
            prefill_chunk_size=256,
            use_inputs_embeds=True,
            final_norm=False,
            exp_silu=True,
            visual={"max_seq_len": [QWENIMAGE21_VISION_PATCHES], "exp_gelu": True},
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLQwenImage21Config",
            sample_sizes=[tuple(size) for size in image_sizes],
            encoder_sample_sizes=condition_sizes,
        )

    @property
    def profile(self) -> str:
        return self.transformer.profile

    @property
    def image_sizes(self) -> list[tuple[int, int]]:
        return self.vae.sample_sizes


__all__ = [
    "RBLNQwenImage21PipelineConfig",
]
