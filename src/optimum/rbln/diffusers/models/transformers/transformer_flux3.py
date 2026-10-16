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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import Flux3Attention, Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep, _prepare_image_ids

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNFlux3Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

# The FLUX.3 pipeline right-pads the prompt to a multiple of this many tokens, and the transformer attends to that
# padding, so the window is part of the model recipe. `max_seq_len` is rounded up to it.
TEXT_PAD_MULTIPLE = 80


def get_image_ids(sample_size: tuple[int, int], image_rope_scale: float) -> torch.Tensor:
    return _prepare_image_ids(sample_size[0], sample_size[1], image_rope_scale)


def get_text_ids(seq_len: int) -> torch.Tensor:
    return Flux3TextRoPEInputsStep._prepare_text_ids(torch.empty(1, seq_len))


def _attention(
    attn: Flux3Attention,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    # Same math as `Flux3AttnProcessor`, with an additive key mask for the text tokens past the prompt window.
    query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

    query = apply_rotary_emb(attn.norm_q(query.float()), rotary_emb, sequence_dim=1).to(value.dtype)
    key = apply_rotary_emb(attn.norm_k(key.float()), rotary_emb, sequence_dim=1).to(value.dtype)

    hidden_states = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
    return attn.to_out[0](hidden_states)


def _modulate(norm: torch.nn.LayerNorm, hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (1 + scale) * norm(hidden_states) + shift


class Flux3Transformer2DModelWrapper(torch.nn.Module):
    """`Flux3Transformer2DModel.forward` for text-to-image as one static graph.

    Differences from the original, all exact up to floating point:

    - The rotary embeddings are compile-time constants: the image grid and the text length are fixed per graph.
    - Every token of a mode shares one timestep embedding, so the modulations are computed on `[batch_size, 1, dim]`
      and broadcast over the tokens instead of being projected per token.
    - The text context is padded to `max_seq_len`; the tokens past the prompt window are masked out of attention.
    - The sinusoidal timestep projections come in as float32 inputs (see `RBLNFlux3Transformer2DModel.forward`).
    """

    def __init__(self, model: Flux3Transformer2DModel, image_ids: torch.Tensor, text_ids: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        with torch.no_grad():
            image_cos, image_sin = model.pos_embed(image_ids)
            text_cos, text_sin = model.pos_embed(text_ids)
            zero_proj = model.time_proj(torch.zeros(1))
        self.register_buffer("image_rotary_emb_cos", image_cos.float(), persistent=False)
        self.register_buffer("image_rotary_emb_sin", image_sin.float(), persistent=False)
        self.register_buffer("joint_rotary_emb_cos", torch.cat([text_cos, image_cos]).float(), persistent=False)
        self.register_buffer("joint_rotary_emb_sin", torch.cat([text_sin, image_sin]).float(), persistent=False)
        self.register_buffer("text_rotary_emb_cos", text_cos.float(), persistent=False)
        self.register_buffer("text_rotary_emb_sin", text_sin.float(), persistent=False)
        self.register_buffer("zero_timestep_proj", zero_proj.float(), persistent=False)

    def _mode_block(self, block, hidden_states, mod, rotary_emb, attention_mask=None):
        shift, scale, gate = mod.chunk(3, dim=-1)
        normed = _modulate(block.norm, hidden_states, shift, scale)
        output = _attention(block.attn, normed, rotary_emb, attention_mask) + block.ff(normed)
        return hidden_states + gate * output

    def _single_block(self, block, hidden_states, encoder_hidden_states, mod, text_mod, rotary_emb, attention_mask):
        shift, scale, gate = mod.chunk(3, dim=-1)
        text_shift, text_scale, text_gate = text_mod.chunk(3, dim=-1)
        combined = torch.cat(
            [
                _modulate(block.norm, encoder_hidden_states, text_shift, text_scale),
                _modulate(block.norm, hidden_states, shift, scale),
            ],
            dim=1,
        )
        output = _attention(block.attn, combined, rotary_emb, attention_mask) + block.ff(combined)
        text_len = encoder_hidden_states.shape[1]
        encoder_hidden_states = encoder_hidden_states + text_gate * output[:, :text_len]
        hidden_states = hidden_states + gate * output[:, text_len:]
        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_proj: torch.Tensor,
        target_timestep_proj: torch.Tensor | None = None,
    ):
        model = self.model
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        # 1. Timestep embeddings: the image tokens at `timestep`, the text context at 0, both plus the target step
        image_temb = model.timestep_embedder(timestep_proj.to(dtype))[:, None, :]
        zero_temb = model.timestep_embedder(self.zero_timestep_proj.to(dtype)).expand(batch_size, -1)[:, None, :]
        if target_timestep_proj is not None:
            target_temb = model.target_timestep_embedder(target_timestep_proj.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb_act = F.silu(image_temb)
        zero_temb_act = F.silu(zero_temb)

        image_early_mod = model.early_stream_modulations["image"](image_temb_act)
        image_single_mod = model.single_stream_modulations["image"](image_temb_act)
        text_early_mod = model.early_stream_modulations["txt"](zero_temb_act)
        text_single_mod = model.single_stream_modulations["txt"](zero_temb_act)

        # 2. Key masks: 0 for the prompt window, -inf-like for the tokens padded to the compiled length
        text_mask = ((encoder_attention_mask - 1) * torch.finfo(torch.float32).max)[:, None, None, :]
        image_mask = torch.zeros(batch_size, 1, 1, hidden_states.shape[1], dtype=text_mask.dtype)
        joint_mask = torch.cat([text_mask, image_mask], dim=-1)

        # 3. Input projections and per-mode blocks
        encoder_hidden_states = model.context_embedder(encoder_hidden_states)
        hidden_states = model.x_embedders["image"](hidden_states)
        text_rotary_emb = (self.text_rotary_emb_cos, self.text_rotary_emb_sin)
        for block in model.txt_mode_blocks:
            encoder_hidden_states = self._mode_block(
                block, encoder_hidden_states, text_early_mod, text_rotary_emb, text_mask
            )
        image_rotary_emb = (self.image_rotary_emb_cos, self.image_rotary_emb_sin)
        for block in model.mode_blocks["image"]:
            hidden_states = self._mode_block(block, hidden_states, image_early_mod, image_rotary_emb)

        # 4. Joint single blocks over [text, image]
        joint_rotary_emb = (self.joint_rotary_emb_cos, self.joint_rotary_emb_sin)
        for block in model.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._single_block(
                block,
                hidden_states,
                encoder_hidden_states,
                image_single_mod,
                text_single_mod,
                joint_rotary_emb,
                joint_mask,
            )

        # 5. Output layers
        hidden_states = model.norm_out(hidden_states, image_temb)
        return (model.proj_out(hidden_states),)


class RBLNFlux3Transformer2DModel(RBLNModel):
    """
    RBLN implementation of Flux3Transformer2DModel for FLUX.3 pipelines such as FLUX 3 Image Klein.

    The whole transformer runs on the RBLN device as one graph compiled for a fixed latent size and a maximum text
    length. Only the sinusoidal projection of the timesteps is computed on the host, in float32, so the timestep keeps
    full precision; the rotary embeddings are constants of the graph.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = Flux3Transformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.time_proj = Timesteps(
            num_channels=self.config.timestep_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.image_ids = get_image_ids(self.rbln_config.sample_size, self.rbln_config.image_rope_scale)
        self.text_ids = get_text_ids(self.rbln_config.max_seq_len)

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNFlux3Transformer2DModelConfig
    ) -> torch.nn.Module:
        image_ids = get_image_ids(rbln_config.sample_size, rbln_config.image_rope_scale)
        text_ids = get_text_ids(rbln_config.max_seq_len)
        return Flux3Transformer2DModelWrapper(model, image_ids, text_ids).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config already sets the latent size and the text length it shares with the text encoder.
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNFlux3Transformer2DModelConfig,
    ) -> RBLNFlux3Transformer2DModelConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = (64, 64)
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = 13 * TEXT_PAD_MULTIPLE
        if rbln_config.max_seq_len % TEXT_PAD_MULTIPLE != 0:
            rbln_config.max_seq_len += TEXT_PAD_MULTIPLE - rbln_config.max_seq_len % TEXT_PAD_MULTIPLE
        if rbln_config.image_rope_scale is None:
            rbln_config.image_rope_scale = 0.5

        batch_size = rbln_config.batch_size
        input_info = [
            (
                "hidden_states",
                [batch_size, rbln_config.sample_size[0] * rbln_config.sample_size[1], model_config.in_channels],
                rbln_config.dtype,
            ),
            (
                "encoder_hidden_states",
                [batch_size, rbln_config.max_seq_len, model_config.joint_attention_dim],
                rbln_config.dtype,
            ),
            ("encoder_attention_mask", [batch_size, rbln_config.max_seq_len], "float32"),
            ("timestep_proj", [batch_size, model_config.timestep_channels], "float32"),
        ]
        if model_config.flow_map:
            input_info.append(("target_timestep_proj", [batch_size, model_config.timestep_channels], "float32"))

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def _check_position_ids(self, img_ids: torch.Tensor, txt_ids: torch.Tensor) -> None:
        # The rotary embeddings are baked into the graph, so the pipeline must ask for the positions it was built for.
        if not torch.equal(img_ids.cpu().float(), self.image_ids):
            raise ValueError(
                "`img_ids` differ from the image grid the transformer was compiled for "
                f"(`sample_size={self.rbln_config.sample_size}`, `image_rope_scale={self.rbln_config.image_rope_scale}`)."
            )
        if not torch.equal(txt_ids.cpu().float(), self.text_ids[: txt_ids.shape[0]]):
            raise ValueError("`txt_ids` differ from the text positions the transformer was compiled for.")

    def _pad_text(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, text_len, _ = encoder_hidden_states.shape
        max_seq_len = self.rbln_config.max_seq_len
        if text_len > max_seq_len:
            raise ValueError(
                f"The text context has {text_len} tokens, but the transformer was compiled for at most "
                f"{max_seq_len} (`max_seq_len`). Recompile with a larger `max_seq_len`."
            )
        pad = max_seq_len - text_len
        mask = torch.cat([torch.ones(batch_size, text_len), torch.zeros(batch_size, pad)], dim=1)
        return F.pad(encoder_hidden_states, (0, 0, 0, pad)), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: torch.Tensor | None = None,
        image_cond_ids: torch.Tensor | None = None,
        target_timestep: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized Flux3Transformer2DModel.

        Args:
            hidden_states (torch.Tensor): Noisy image latent tokens of shape `(batch_size, image_seq_len, in_channels)`.
            encoder_hidden_states (torch.Tensor): Text context embeddings of shape
                `(batch_size, text_seq_len, joint_attention_dim)` (stacked Qwen3-VL hidden states).
            timestep (torch.Tensor): Flow timestep in `[0, 1]` of the image tokens, of shape `(batch_size,)`.
            img_ids (torch.Tensor): Position ids `(t, h, w, l)` of the image tokens, of shape `(image_seq_len, 4)`.
                They must match the image grid the model was compiled for.
            txt_ids (torch.Tensor): Position ids of the text context tokens, of shape `(text_seq_len, 4)`.
            image_cond_hidden_states (torch.Tensor | None): Reference image tokens. Not supported yet.
            image_cond_ids (torch.Tensor | None): Position ids of the reference image tokens. Not supported yet.
            target_timestep (torch.Tensor | None): The timestep the current step jumps to. Required by, and only by,
                a flow-map (step-distilled) model.
            attention_kwargs (dict | None): Unused; accepted for compatibility with the diffusers signature.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_output.Transformer2DModelOutput` | tuple)
        """
        if image_cond_hidden_states is not None:
            raise NotImplementedError("Reference images are not supported by RBLNFlux3Transformer2DModel yet.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError(
                "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
            )
        dtype = self.rbln_config.dtype
        encoder_hidden_states, encoder_attention_mask = self._pad_text(encoder_hidden_states)
        self._check_position_ids(img_ids, txt_ids)
        inputs = [
            hidden_states.to(dtype),
            encoder_hidden_states.to(dtype),
            encoder_attention_mask,
            self.time_proj(timestep.float() * 1000).float(),
        ]
        if target_timestep is not None:
            inputs.append(self.time_proj(target_timestep.float() * 1000).float())

        output = self.model[0](*inputs)
        return self._prepare_output(output, return_dict)


__all__ = [
    "RBLNFlux3Transformer2DModel",
]
