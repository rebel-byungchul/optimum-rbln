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

import PIL.Image
import PIL.ImageOps
from diffusers import Flux3DistilledModularPipeline, Flux3ModularPipeline
from diffusers.modular_pipelines.flux3.prompt_upsampler import MAX_PIXELS_BY_REFS, MAX_REFERENCES

from ...configurations.pipelines.configuration_flux3 import (
    RBLNFlux3DistilledModularPipelineConfig,
    RBLNFlux3ModularPipelineConfig,
)
from ...modeling_diffusers import RBLNModularPipelineMixin


def reference_bucket(
    image: PIL.Image.Image, num_references: int, encoder_sample_sizes: list[tuple[int, int]]
) -> tuple[int, int]:
    """The compiled (height, width) a reference image is resized to: among the sizes for this number of references (the
    prompt upsampler's pixel budget for it), the one whose aspect ratio is closest to the image's."""
    budget = MAX_PIXELS_BY_REFS[num_references]
    floor = max((b for b in MAX_PIXELS_BY_REFS.values() if b < budget), default=0)
    candidates = [size for size in encoder_sample_sizes if floor < size[0] * size[1] <= budget]
    if not candidates:
        raise ValueError(
            f"The VAE encoder was not compiled for {num_references} reference image(s) "
            "(see `max_references` in the pipeline's RBLN config)."
        )
    ratio = math.log(image.width / image.height)
    return min(candidates, key=lambda size: abs(math.log(size[1] / size[0]) - ratio))


class RBLNFlux3ModularPipeline(RBLNModularPipelineMixin, Flux3ModularPipeline):
    """
    RBLN-accelerated FLUX.3 modular pipeline (FLUX 3 Image Klein base: 50 steps, classifier-free guidance).

    The prompt upsampler (a Qwen3.5 model that expands the request into the full caption), the Qwen3-VL text encoder,
    the transformer and the VAE run on RBLN NPUs, compiled for one image size and a maximum prompt length (see
    [`RBLNFlux3ModularPipelineConfig`]). Classifier-free guidance runs the transformer separately for the prompt and
    the negative prompt, so the guider and `negative_prompt` work as in diffusers.

    Reference images (`image`, up to 10) are center-cropped to the closest compiled aspect ratio and resized to the size
    the prompt upsampler uses for that many references (1024x1024 pixels for one, 724x724 for two, 512x512 for three or
    more), since the VAE encoder and the transformer are compiled for those sizes. The diffusers pipeline instead keeps
    a single reference whole up to 4 MP, padded to the 16-pixel grid and generated at its size, and fits each of several
    references under 1 MP. So here a reference larger than about 1 MP, or each of several references, conditions the
    generation at a lower resolution (two 1024x1024 references at 704x704 each), a reference whose aspect ratio falls
    between the compiled ones loses up to about 13 % of one side to the crop, and the generated image keeps the compiled
    size rather than the reference's.

    Example:
        ```python
        from optimum.rbln import RBLNFlux3ModularPipeline

        pipe = RBLNFlux3ModularPipeline.from_pretrained(
            "black-forest-labs/flux-3-image-klein-base", export=True, model_save_dir="flux3-klein-base-rbln"
        )
        image = pipe(prompt="A red fox in a snow-covered forest clearing at golden hour", output="images")[0]
        edited = pipe(prompt="the same scene at night under a full moon", image=image, output="images")[0]
        ```
    """

    original_class = Flux3ModularPipeline
    _rbln_config_class = RBLNFlux3ModularPipelineConfig
    _submodules = ["prompt_upsampler", "text_encoder", "transformer", "vae"]
    _upcasts_vae = True

    @classmethod
    def _compile_submodules(cls, model, passed_submodules, model_save_dir, rbln_config, prefix=""):
        # The text window, the latent patch, the encoder layers and the rotary scale come from the diffusers pipeline.
        rbln_config.update_using_pipe(model)
        return super()._compile_submodules(model, passed_submodules, model_save_dir, rbln_config, prefix=prefix)

    def handle_additional_kwargs(self, **kwargs):
        kwargs = super().handle_additional_kwargs(**kwargs)
        # Prompts are truncated at the compiled length, as the pipeline truncates at `max_sequence_length`.
        if kwargs.get("max_sequence_length") is None:
            kwargs["max_sequence_length"] = self.text_encoder.rbln_config.max_seq_len
        if kwargs.get("image") is not None:
            kwargs["image"] = self._fit_references(kwargs["image"])
        return kwargs

    def _fit_references(self, image: PIL.Image.Image | list[PIL.Image.Image]) -> PIL.Image.Image | list:
        images = [image] if isinstance(image, PIL.Image.Image) else list(image)
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"FLUX.3 takes at most {MAX_REFERENCES} reference images, got {len(images)}.")
        encoder_sample_sizes = self.vae.rbln_config.encoder_sample_sizes or []
        fitted = []
        for reference in images:
            height, width = reference_bucket(reference, len(images), encoder_sample_sizes)
            fitted.append(PIL.ImageOps.fit(reference, (width, height), PIL.Image.Resampling.LANCZOS))
        return fitted[0] if isinstance(image, PIL.Image.Image) else fitted


class RBLNFlux3DistilledModularPipeline(RBLNFlux3ModularPipeline, Flux3DistilledModularPipeline):
    """
    RBLN-accelerated step-distilled FLUX.3 modular pipeline (FLUX 3 Image Klein: 16 steps, no guidance).

    It runs like [`RBLNFlux3ModularPipeline`], whose documentation covers the compiled submodules and the reference
    images, with the step-distilled blocks of diffusers.

    Example:
        ```python
        from optimum.rbln import RBLNFlux3DistilledModularPipeline

        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(
            "black-forest-labs/flux-3-image-klein", export=True, model_save_dir="flux3-klein-rbln"
        )
        image = pipe(prompt="A red fox in a snow-covered forest clearing at golden hour", output="images")[0]
        ```
    """

    original_class = Flux3DistilledModularPipeline
    _rbln_config_class = RBLNFlux3DistilledModularPipelineConfig


__all__ = [
    "RBLNFlux3DistilledModularPipeline",
    "RBLNFlux3ModularPipeline",
]
