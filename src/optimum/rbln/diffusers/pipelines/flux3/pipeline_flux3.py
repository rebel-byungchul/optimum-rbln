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

from diffusers import Flux3DistilledModularPipeline

from ...configurations.pipelines.configuration_flux3 import RBLNFlux3DistilledModularPipelineConfig
from ...modeling_diffusers import RBLNModularPipelineMixin


class RBLNFlux3DistilledModularPipeline(RBLNModularPipelineMixin, Flux3DistilledModularPipeline):
    """
    RBLN-accelerated step-distilled FLUX.3 modular pipeline (FLUX 3 Image Klein) for text-to-image generation.

    The prompt upsampler (a Qwen3.5 model that expands the request into the full caption), the Qwen3-VL text encoder,
    the transformer and the VAE decoder run on RBLN NPUs, compiled for one image size and a maximum prompt length
    (see [`RBLNFlux3DistilledModularPipelineConfig`]). Reference images are not supported yet.

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
    _submodules = ["prompt_upsampler", "text_encoder", "transformer", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        kwargs = super().handle_additional_kwargs(**kwargs)
        # Prompts are truncated at the compiled length, as the pipeline truncates at `max_sequence_length`.
        if kwargs.get("max_sequence_length") is None:
            kwargs["max_sequence_length"] = self.text_encoder.rbln_config.max_seq_len
        return kwargs


__all__ = [
    "RBLNFlux3DistilledModularPipeline",
]
