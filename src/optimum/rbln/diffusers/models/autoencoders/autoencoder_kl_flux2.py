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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import rebel
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling_base import Preprocessor
from ...configurations import RBLNAutoencoderKLFlux2Config
from .autoencoder_kl import RBLNAutoencoderKL, _PostQuantConv
from .vae import RBLNRuntimeVAEDecoder, RBLNRuntimeVAEEncoder, _VAEDecoder, _VAEEncoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


def _encoder_graph_name(size: tuple[int, int]) -> str:
    return f"encoder_{size[0]}x{size[1]}"


class RBLNAutoencoderKLFlux2(RBLNAutoencoderKL):
    """
    RBLN implementation of AutoencoderKLFlux2, the VAE of FLUX.2 and FLUX.3 pipelines.

    The decoder is compiled like [`RBLNAutoencoderKL`]'s for `sample_size`, and the encoder once per size in
    `encoder_sample_sizes` (FLUX pipelines encode reference images of their own sizes); `encode` runs the graph of its
    input's size. FLUX pipelines normalize the packed latents with the VAE's batch-norm running statistics (`vae.bn`)
    outside `encode`/`decode`, so those statistics are kept on the host and exposed as `bn`.

    This class inherits from [`RBLNAutoencoderKL`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLFlux2
    _rbln_config_class = RBLNAutoencoderKLFlux2Config

    def __post_init__(self, **kwargs):
        # Not RBLNAutoencoderKL.__post_init__: the graphs are looked up by name, there can be several encoders.
        super(RBLNAutoencoderKL, self).__post_init__(**kwargs)
        # Indexed rather than zipped: without runtimes (create_runtimes=False) `self.model` is an UnavailableRuntime.
        graphs = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        self.decoder = RBLNRuntimeVAEDecoder(runtime=graphs.pop("decoder"), main_input_name="z")
        self.encoders = {
            size: RBLNRuntimeVAEEncoder(runtime=graphs[_encoder_graph_name(size)], main_input_name="x")
            for size in self.rbln_config.encoder_sample_sizes or []
        }
        self.encoder = None
        self.image_size = self.rbln_config.image_size
        self.post_quant_conv = _PostQuantConv(self.dtype)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.config.patch_size) * self.config.latent_channels,
            eps=self.config.batch_norm_eps,
            momentum=self.config.batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        ).eval()
        self.bn.load_state_dict(artifacts["bn"])

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # FLUX pipeline configs set the image size; the transformer has no `sample_size` to derive it from.
        if getattr(rbln_config, submodule_name).sample_size is None:
            raise ValueError(
                f"The pipeline config must set `{submodule_name}.sample_size` for RBLNAutoencoderKLFlux2."
            )
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLFlux2Config,
    ) -> RBLNAutoencoderKLFlux2Config:
        # The parent sets the decoder (and the sizes); the encoder graphs are added per size here.
        uses_encoder, rbln_config.uses_encoder = rbln_config.uses_encoder, False
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)
        rbln_config.uses_encoder = uses_encoder
        if uses_encoder and rbln_config.encoder_sample_sizes is None:
            rbln_config.encoder_sample_sizes = [tuple(rbln_config.sample_size)]

        encoder_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=_encoder_graph_name(size),
                input_info=[("x", [rbln_config.batch_size, rbln_config.in_channels, *size], rbln_config.dtype)],
            )
            for size in rbln_config.encoder_sample_sizes or []
        ]
        rbln_config.set_compile_cfgs(encoder_cfgs + rbln_config.compile_cfgs)
        return rbln_config

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLFlux2Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            name = compile_config.compiled_model_name
            wrapped_model = _VAEDecoder(model) if name == "decoder" else _VAEEncoder(model)
            compiled_models[name] = cls.compile(
                wrapped_model.eval(),
                rbln_compile_config=compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
            )
        return compiled_models

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNAutoencoderKLFlux2Config
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if any(name not in rbln_config.device_map for name in names):
            cls._raise_missing_compiled_file_error(names)
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for name, compiled_model in zip(names, compiled_models, strict=True)
        ]

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: dict[str, Any]
    ) -> torch.FloatTensor | AutoencoderKLOutput:
        """
        Encode an input image into a latent representation, with the encoder graph compiled for its size.

        Args:
            x: The input image to encode, of shape `(batch_size, 3, height, width)`.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the encoder.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        size = tuple(x.shape[-2:])
        if size not in self.encoders:
            raise ValueError(
                f"No encoder was compiled for {size[0]}x{size[1]} images; compiled sizes: "
                f"{sorted(self.encoders)} (`encoder_sample_sizes`)."
            )
        posterior = self.encoders[size].encode(x.to(self.rbln_config.dtype))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        torch.save({"bn": model.bn.state_dict()}, save_dir_path / subfolder / "torch_artifacts.pth")


__all__ = [
    "RBLNAutoencoderKLFlux2",
]
