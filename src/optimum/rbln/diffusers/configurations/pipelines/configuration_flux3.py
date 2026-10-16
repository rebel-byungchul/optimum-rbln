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

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3_5ForConditionalGenerationConfig, RBLNQwen3VLTextEncoderConfig
from ..models import RBLNAutoencoderKLFlux2Config, RBLNFlux3Transformer2DModelConfig


# The Qwen3-VL hidden states the FLUX.3 text encoder step stacks (its `text_encoder_out_layers` default).
FLUX3_TEXT_ENCODER_OUT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
# The FLUX.3 pipeline right-pads the prompt to a multiple of this many tokens.
FLUX3_TEXT_PAD_MULTIPLE = 80
# The packed FLUX.2 latents are 16 times smaller than the image: an 8x VAE and a 2x2 patch.
FLUX3_IMAGE_TO_LATENT = 16


class RBLNFlux3DistilledModularPipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN step-distilled FLUX.3 modular pipeline (FLUX 3 Image Klein).

    The image size and the maximum prompt length are fixed at compile time and shared by the submodules: the text
    encoder and the transformer are compiled for the same `max_seq_len`, and the transformer and the VAE for the same
    image size. The prompt upsampler, which writes the caption the text encoder reads, has its own length.
    """

    submodules = ["prompt_upsampler", "text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3VLTextEncoderConfig | None = None,
        transformer: RBLNFlux3Transformer2DModelConfig | None = None,
        vae: RBLNAutoencoderKLFlux2Config | None = None,
        prompt_upsampler: RBLNQwen3_5ForConditionalGenerationConfig | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
        max_seq_len: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3VLTextEncoderConfig | None): Configuration for the Qwen3-VL text encoder.
                Initialized as RBLNQwen3VLTextEncoderConfig if not provided.
            transformer (RBLNFlux3Transformer2DModelConfig | None): Configuration for the transformer.
                Initialized as RBLNFlux3Transformer2DModelConfig if not provided.
            vae (RBLNAutoencoderKLFlux2Config | None): Configuration for the VAE (decoder only).
                Initialized as RBLNAutoencoderKLFlux2Config if not provided.
            prompt_upsampler (RBLNQwen3_5ForConditionalGenerationConfig | None): Configuration for the prompt
                upsampler. Initialized as RBLNQwen3_5ForConditionalGenerationConfig if not provided, with room for
                the system prompt and the step's 3584 new tokens (`max_seq_len` 8192, flash attention with
                `kvcache_partition_len` 4096).
            height (int | None): Height of the generated images, a multiple of 16. Defaults to 1024.
            width (int | None): Width of the generated images, a multiple of 16. Defaults to 1024.
            max_seq_len (int | None): Maximum prompt length in tokens (after the chat template), for the text
                encoder and the transformer. Rounded up to a multiple of 80. Defaults to 1040.
            batch_size (int | None): Batch size for inference, applied to all submodules. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If only one of height and width is given, or if they are not multiples of 16.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")
        height, width = height or 1024, width or 1024
        if height % FLUX3_IMAGE_TO_LATENT or width % FLUX3_IMAGE_TO_LATENT:
            raise ValueError(f"height and width must be multiples of {FLUX3_IMAGE_TO_LATENT}, got {height}x{width}")
        max_seq_len = max_seq_len or 1040
        max_seq_len = -(-max_seq_len // FLUX3_TEXT_PAD_MULTIPLE) * FLUX3_TEXT_PAD_MULTIPLE

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3VLTextEncoderConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            hidden_states_layers=FLUX3_TEXT_ENCODER_OUT_LAYERS,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNFlux3Transformer2DModelConfig",
            batch_size=batch_size,
            sample_size=(height // FLUX3_IMAGE_TO_LATENT, width // FLUX3_IMAGE_TO_LATENT),
            max_seq_len=max_seq_len,
        )
        self.prompt_upsampler = self.initialize_submodule_config(
            prompt_upsampler,
            cls_name="RBLNQwen3_5ForConditionalGenerationConfig",
            batch_size=1,
            max_seq_len=8192,
            # Flash attention, as upstream tests Qwen3.5: on RBLN-CR13 (rebel-compiler 0.12.1.dev163) the eager
            # attention build of this model generates garbage after the first token.
            kvcache_partition_len=4096,
            visual={"max_seq_len": 4096},
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLFlux2Config",
            batch_size=batch_size,
            uses_encoder=False,
            sample_size=(height, width),
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return self.vae.sample_size


__all__ = [
    "RBLNFlux3DistilledModularPipelineConfig",
]
