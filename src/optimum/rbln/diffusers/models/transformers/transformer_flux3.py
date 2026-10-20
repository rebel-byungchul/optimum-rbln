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

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import rebel
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import Flux3Attention, Flux3PosEmbed, Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.before_denoise import (
    Flux3PrepareLatentsStep,
    Flux3TextRoPEInputsStep,
    _prepare_image_ids,
)
from rebel.compile_context import CompileContext

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNFlux3Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)


def get_image_ids(sample_size: tuple[int, int], image_rope_scale: float) -> torch.Tensor:
    return _prepare_image_ids(sample_size[0], sample_size[1], image_rope_scale)


def get_text_ids(seq_len: int) -> torch.Tensor:
    return Flux3TextRoPEInputsStep._prepare_text_ids(torch.empty(1, seq_len))


def graph_name(num_reference_tokens: int, num_text_tokens: int) -> str:
    if num_reference_tokens:
        return f"image_conditioned_{num_reference_tokens}_{num_text_tokens}"
    return f"text2image_{num_text_tokens}"


def text_buckets_for(rbln_config: RBLNFlux3Transformer2DModelConfig, num_reference_tokens: int) -> list[int]:
    """The text lengths compiled for a graph: all of them, but `max_seq_len` alone for the image-conditioned graph of
    the largest of several reference capacities, whose working memory is the largest."""
    buckets = rbln_config.reference_token_buckets
    if len(buckets) > 1 and num_reference_tokens == buckets[-1]:
        return [rbln_config.max_seq_len]
    return rbln_config.text_token_buckets


def graph_specs(rbln_config: RBLNFlux3Transformer2DModelConfig) -> list[tuple[int, int]]:
    """The (reference tokens, text tokens) of every graph, text-to-image ones (0 reference tokens) first."""
    return [
        (num_reference_tokens, num_text_tokens)
        for num_reference_tokens in [0, *rbln_config.reference_token_buckets]
        for num_text_tokens in text_buckets_for(rbln_config, num_reference_tokens)
    ]


def graph_input_names(flow_map: bool, with_references: bool) -> list[str]:
    """The inputs of a transformer graph, in order."""
    names = ["hidden_states", "encoder_hidden_states", "encoder_attention_mask", "timestep_proj"]
    if flow_map:
        names.append("target_timestep_proj")
    if with_references:
        names += [
            "image_cond_hidden_states",
            "image_cond_attention_mask",
            "image_cond_rotary_emb_cos",
            "image_cond_rotary_emb_sin",
        ]
    return names


def _key_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # 1 -> 0 (attend), 0 -> a large negative bias, shaped [batch_size, 1, 1, seq_len] for all heads and queries
    return ((attention_mask - 1) * torch.finfo(torch.float32).max)[:, None, None, :]


def _attention(
    attn: Flux3Attention,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    query_chunk_size: int | None = None,
) -> torch.Tensor:
    # Same math as `Flux3AttnProcessor`, with an additive key mask for the tokens padded to the compiled lengths.
    query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

    query = apply_rotary_emb(attn.norm_q(query.float()), rotary_emb, sequence_dim=1).to(value.dtype).transpose(1, 2)
    key = apply_rotary_emb(attn.norm_k(key.float()), rotary_emb, sequence_dim=1).to(value.dtype).transpose(1, 2)
    value = value.transpose(1, 2)

    # Attending in query chunks bounds the attention scores the graph holds at once to
    # [heads, query_chunk_size, seq_len] instead of [heads, seq_len, seq_len].
    query_chunks = query.split(query_chunk_size, dim=2) if query_chunk_size else [query]
    hidden_states = torch.cat(
        [F.scaled_dot_product_attention(q, key, value, attn_mask=attention_mask) for q in query_chunks], dim=2
    )
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
    return attn.to_out[0](hidden_states)


def _modulate(norm: torch.nn.LayerNorm, hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (1 + scale) * norm(hidden_states) + shift


class Flux3Transformer2DModelWrapper(torch.nn.Module):
    """`Flux3Transformer2DModel.forward` as one static graph, for text-to-image or with reference images.

    Differences from the original, all exact up to floating point:

    - The rotary embeddings of the image and text tokens are compile-time constants: the image grid and the text length
      are fixed per graph. Those of the reference tokens depend on the references and come in as inputs.
    - Every token of a mode shares one timestep embedding, so the modulations are computed on `[batch_size, 1, dim]`
      and broadcast over the tokens instead of being projected per token.
    - The text context is padded to `num_text_tokens` and the reference tokens to `num_reference_tokens`; the padded
      tokens are masked out of attention.
    - The sinusoidal timestep projections come in as float32 inputs (see `RBLNFlux3Transformer2DModel.forward`).
    - Attention runs in query chunks of `query_chunk_size`, which bounds the attention scores held at once.

    The inputs are positional, in the order of `graph_input_names`. One wrapper serves every graph: set
    `num_reference_tokens` (0 for text-to-image) and `num_text_tokens` before tracing each. Graphs traced from one module share its weights
    under a weight-sharing compile context; a module per graph, each with its own constant buffers, would not.
    """

    def __init__(
        self,
        model: Flux3Transformer2DModel,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        text_token_buckets: list[int],
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_reference_tokens = 0
        self.num_text_tokens = text_token_buckets[-1]
        self.query_chunk_size = query_chunk_size
        with torch.no_grad():
            image_cos, image_sin = model.pos_embed(image_ids)
            text_cos, text_sin = model.pos_embed(text_ids)
            zero_proj = model.time_proj(torch.zeros(1))
        self.register_buffer("image_rotary_emb_cos", image_cos.float(), persistent=False)
        self.register_buffer("image_rotary_emb_sin", image_sin.float(), persistent=False)
        # One set per text length, built here rather than in forward: concatenating or slicing constants in a graph
        # compiled with weight sharing crashes rebel-compiler 0.12.1.dev163 (a constant with an input is fine).
        for length in text_token_buckets:
            cos, sin = text_cos[:length].float(), text_sin[:length].float()
            self.register_buffer(f"text_rotary_emb_cos_{length}", cos, persistent=False)
            self.register_buffer(f"text_rotary_emb_sin_{length}", sin, persistent=False)
            self.register_buffer(
                f"joint_rotary_emb_cos_{length}", torch.cat([cos, image_cos.float()]), persistent=False
            )
            self.register_buffer(
                f"joint_rotary_emb_sin_{length}", torch.cat([sin, image_sin.float()]), persistent=False
            )
        self.register_buffer("zero_timestep_proj", zero_proj.float(), persistent=False)

    def _rotary_emb(self, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        length = self.num_text_tokens
        return getattr(self, f"{kind}_rotary_emb_cos_{length}"), getattr(self, f"{kind}_rotary_emb_sin_{length}")

    def _mode_block(self, block, hidden_states, mod, rotary_emb, attention_mask=None):
        shift, scale, gate = mod.chunk(3, dim=-1)
        normed = _modulate(block.norm, hidden_states, shift, scale)
        output = _attention(block.attn, normed, rotary_emb, attention_mask, self.query_chunk_size) + block.ff(normed)
        return hidden_states + gate * output

    def _single_block(self, block, streams, mods, rotary_emb, attention_mask):
        # `streams` are the text context and the content modes, each with its own modulation, attending jointly
        modulations = [mod.chunk(3, dim=-1) for mod in mods]
        combined = torch.cat(
            [
                _modulate(block.norm, x, shift, scale)
                for x, (shift, scale, _) in zip(streams, modulations, strict=True)
            ],
            dim=1,
        )
        output = _attention(block.attn, combined, rotary_emb, attention_mask, self.query_chunk_size) + block.ff(
            combined
        )
        outputs = output.split([x.shape[1] for x in streams], dim=1)
        return [x + gate * out for x, (_, _, gate), out in zip(streams, modulations, outputs, strict=True)]

    def forward(self, *inputs: torch.Tensor):
        input_names = graph_input_names(bool(self.model.config.flow_map), self.num_reference_tokens > 0)
        args = dict(zip(input_names, inputs, strict=True))
        model = self.model
        hidden_states = args["hidden_states"]
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        # 1. Timestep embeddings: the image tokens at `timestep`, the text context and the references at 0, all plus
        #    the embedding of the target step for a flow-map model
        image_temb = model.timestep_embedder(args["timestep_proj"].to(dtype))[:, None, :]
        zero_temb = model.timestep_embedder(self.zero_timestep_proj.to(dtype)).expand(batch_size, -1)[:, None, :]
        if "target_timestep_proj" in args:
            target_temb = model.target_timestep_embedder(args["target_timestep_proj"].to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb_act = F.silu(image_temb)
        zero_temb_act = F.silu(zero_temb)

        # 2. Input projections, key masks and the per-mode blocks of the text and the image
        text = model.context_embedder(args["encoder_hidden_states"])
        image = model.x_embedders["image"](hidden_states)
        text_mask = _key_mask(args["encoder_attention_mask"])
        text_rotary_emb = self._rotary_emb("text")
        text_early_mod = model.early_stream_modulations["txt"](zero_temb_act)
        for block in model.txt_mode_blocks:
            text = self._mode_block(block, text, text_early_mod, text_rotary_emb, text_mask)
        image_early_mod = model.early_stream_modulations["image"](image_temb_act)
        image_rotary_emb = (self.image_rotary_emb_cos, self.image_rotary_emb_sin)
        for block in model.mode_blocks["image"]:
            image = self._mode_block(block, image, image_early_mod, image_rotary_emb)

        streams = [text, image]
        mods = [
            model.single_stream_modulations["txt"](zero_temb_act),
            model.single_stream_modulations["image"](image_temb_act),
        ]
        masks = [text_mask, torch.zeros(batch_size, 1, 1, image.shape[1], dtype=text_mask.dtype)]
        joint_rotary_emb = self._rotary_emb("joint")

        # 3. Reference tokens: their own per-mode blocks, then the joint blocks after the image tokens
        if self.num_reference_tokens:
            cond = model.x_embedders["image_cond"](args["image_cond_hidden_states"])
            cond_mask = _key_mask(args["image_cond_attention_mask"])
            cond_rotary_emb = (args["image_cond_rotary_emb_cos"], args["image_cond_rotary_emb_sin"])
            cond_early_mod = model.early_stream_modulations["image_cond"](zero_temb_act)
            for block in model.mode_blocks["image_cond"]:
                cond = self._mode_block(block, cond, cond_early_mod, cond_rotary_emb, cond_mask)
            streams.append(cond)
            mods.append(model.single_stream_modulations["image_cond"](zero_temb_act))
            masks.append(cond_mask)
            joint_rotary_emb = tuple(
                torch.cat([joint, cond]) for joint, cond in zip(joint_rotary_emb, cond_rotary_emb, strict=True)
            )

        # 4. Joint single blocks over [text, image, references]
        joint_mask = torch.cat(masks, dim=-1)
        for block in model.single_transformer_blocks:
            streams = self._single_block(block, streams, mods, joint_rotary_emb, joint_mask)

        # 5. Output layers, on the image tokens only
        image = model.norm_out(streams[1], image_temb)
        return (model.proj_out(image),)


class RBLNFlux3Transformer2DModel(RBLNModel):
    """
    RBLN implementation of Flux3Transformer2DModel for FLUX.3 pipelines such as FLUX 3 Image Klein.

    The transformer runs on the RBLN device as static graphs compiled for a fixed latent size: for text-to-image and,
    when `reference_token_buckets` is set, per reference-token capacity for image-conditioned generation, each at the
    text lengths of `text_token_buckets`. A call runs the smallest graph that holds its text and reference tokens. The
    graphs share their weights unless `share_weights` is off. Only the sinusoidal projection of the timesteps and the
    rotary embeddings of the reference tokens are computed on the host, in float32; the rotary embeddings of the image
    and text tokens are constants of the graphs.

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
        self.pos_embed = Flux3PosEmbed(theta=self.config.rope_theta, axes_dim=list(self.config.axes_dims_rope))
        self.image_ids = get_image_ids(self.rbln_config.sample_size, self.rbln_config.image_rope_scale)
        self.text_ids = get_text_ids(self.rbln_config.max_seq_len)
        # Indexed rather than zipped: without runtimes (create_runtimes=False) `self.model` is an UnavailableRuntime.
        self.graphs = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The FLUX.3 pipeline fills in the latent size, the text lengths and the rotary scale from its own blocks.
        return rbln_config

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNFlux3Transformer2DModelConfig
    ) -> torch.nn.Module:
        image_ids = get_image_ids(rbln_config.sample_size, rbln_config.image_rope_scale)
        text_ids = get_text_ids(rbln_config.max_seq_len)
        return Flux3Transformer2DModelWrapper(
            model, image_ids, text_ids, rbln_config.text_token_buckets, rbln_config.attention_query_chunk_size
        ).eval()

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNFlux3Transformer2DModelConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        # One wrapper and one compile context for all graphs, so that they can share the transformer weights.
        wrapped_model = cls._wrap_model_if_needed(model, rbln_config)
        context = CompileContext(use_weight_sharing=rbln_config.share_weights)
        specs = {graph_name(*spec): spec for spec in graph_specs(rbln_config)}
        cfgs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        # The image-conditioned graphs are compiled first: under weight sharing the weights are laid out for the
        # first graph compiled, and on RBLN-CR13 that layout leaves the text-to-image graphs as fast as before but
        # speeds up the image-conditioned graphs (517 -> 418 ms per step with 4096 reference tokens).
        compiled_models = {}
        for name in sorted(specs, key=lambda name: (specs[name][0] == 0, specs[name])):
            wrapped_model.num_reference_tokens, wrapped_model.num_text_tokens = specs[name]
            compiled_models[name] = cls.compile(
                wrapped_model,
                rbln_compile_config=cfgs[name],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
                compile_context=context,
            )
        return {name: compiled_models[name] for name in specs}

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNFlux3Transformer2DModelConfig
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
            rbln_config.max_seq_len = 1040
        buckets = {n for n in rbln_config.text_token_buckets or [] if n <= rbln_config.max_seq_len}
        rbln_config.text_token_buckets = sorted(buckets | {rbln_config.max_seq_len})
        if rbln_config.image_rope_scale is None:
            signature = inspect.signature(Flux3PrepareLatentsStep.__init__)
            rbln_config.image_rope_scale = signature.parameters["image_rope_scale"].default
        if rbln_config.attention_query_chunk_size is None:
            rbln_config.attention_query_chunk_size = 1024

        batch_size = rbln_config.batch_size
        head_dim = model_config.attention_head_dim
        shapes = {
            "hidden_states": (
                [batch_size, rbln_config.sample_size[0] * rbln_config.sample_size[1], model_config.in_channels],
                rbln_config.dtype,
            ),
            "timestep_proj": ([batch_size, model_config.timestep_channels], "float32"),
            "target_timestep_proj": ([batch_size, model_config.timestep_channels], "float32"),
        }

        def compile_config(num_reference_tokens, num_text_tokens):
            reference_shapes = {
                "encoder_hidden_states": (
                    [batch_size, num_text_tokens, model_config.joint_attention_dim],
                    rbln_config.dtype,
                ),
                "encoder_attention_mask": ([batch_size, num_text_tokens], "float32"),
                "image_cond_hidden_states": (
                    [batch_size, num_reference_tokens, model_config.cond_in_channels],
                    rbln_config.dtype,
                ),
                "image_cond_attention_mask": ([batch_size, num_reference_tokens], "float32"),
                "image_cond_rotary_emb_cos": ([num_reference_tokens, head_dim], "float32"),
                "image_cond_rotary_emb_sin": ([num_reference_tokens, head_dim], "float32"),
            }
            input_names = graph_input_names(bool(model_config.flow_map), num_reference_tokens > 0)
            all_shapes = shapes | reference_shapes
            return RBLNCompileConfig(
                compiled_model_name=graph_name(num_reference_tokens, num_text_tokens),
                input_info=[(n, *all_shapes[n]) for n in input_names],
            )

        rbln_config.set_compile_cfgs([compile_config(*spec) for spec in graph_specs(rbln_config)])
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

    @staticmethod
    def _pad_tokens(tokens: torch.Tensor, length: int, what: str, option: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Pads `[batch_size, seq_len, ...]` to `length` tokens and returns it with its 1/0 key mask."""
        batch_size, seq_len = tokens.shape[:2]
        if seq_len > length:
            raise ValueError(
                f"The {what} has {seq_len} tokens, but the transformer was compiled for at most {length} (`{option}`)."
            )
        mask = torch.cat([torch.ones(batch_size, seq_len), torch.zeros(batch_size, length - seq_len)], dim=1)
        return F.pad(tokens, (0, 0) * (tokens.dim() - 2) + (0, length - seq_len)), mask

    def _reference_inputs(self, image_cond_hidden_states: torch.Tensor, image_cond_ids: torch.Tensor) -> tuple:
        buckets = [n for n in self.rbln_config.reference_token_buckets if n >= image_cond_hidden_states.shape[1]]
        if not buckets:
            raise ValueError(
                f"The reference images have {image_cond_hidden_states.shape[1]} tokens, but the transformer was "
                f"compiled for at most {max(self.rbln_config.reference_token_buckets or [0])} "
                "(`reference_token_buckets`)."
            )
        num_reference_tokens = min(buckets)
        tokens, mask = self._pad_tokens(
            image_cond_hidden_states, num_reference_tokens, "reference images", "reference_token_buckets"
        )
        # The positions of the reference tokens depend on the references, so their rotary embeddings are computed here.
        cos, sin = (
            F.pad(x.float(), (0, 0, 0, num_reference_tokens - x.shape[0])) for x in self.pos_embed(image_cond_ids)
        )
        return num_reference_tokens, [tokens.to(self.rbln_config.dtype), mask, cos, sin]

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
                `(batch_size, text_seq_len, joint_attention_dim)` (stacked Qwen3-VL hidden states). Runs the graph of
                the smallest text bucket that holds them.
            timestep (torch.Tensor): Flow timestep in `[0, 1]` of the image tokens, of shape `(batch_size,)`.
            img_ids (torch.Tensor): Position ids `(t, h, w, l)` of the image tokens, of shape `(image_seq_len, 4)`.
                They must match the image grid the model was compiled for.
            txt_ids (torch.Tensor): Position ids of the text context tokens, of shape `(text_seq_len, 4)`.
            image_cond_hidden_states (torch.Tensor | None): Reference image tokens of shape
                `(batch_size, cond_seq_len, cond_in_channels)`. Runs the smallest image-conditioned graph that holds
                them.
            image_cond_ids (torch.Tensor | None): Position ids of the reference image tokens, of shape
                `(cond_seq_len, 4)`. Required with `image_cond_hidden_states`.
            target_timestep (torch.Tensor | None): The timestep the current step jumps to. Required by, and only by,
                a flow-map (step-distilled) model.
            attention_kwargs (dict | None): Unused; accepted for compatibility with the diffusers signature.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_output.Transformer2DModelOutput` | tuple)
        """
        if (image_cond_hidden_states is None) != (image_cond_ids is None):
            raise ValueError("`image_cond_hidden_states` and `image_cond_ids` must be passed together.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError(
                "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
            )

        dtype = self.rbln_config.dtype
        self._check_position_ids(img_ids, txt_ids)
        num_reference_tokens, reference_inputs = 0, []
        if image_cond_hidden_states is not None:
            num_reference_tokens, reference_inputs = self._reference_inputs(image_cond_hidden_states, image_cond_ids)
        text_buckets = text_buckets_for(self.rbln_config, num_reference_tokens)
        num_text_tokens = next((n for n in text_buckets if n >= encoder_hidden_states.shape[1]), text_buckets[-1])
        encoder_hidden_states, encoder_attention_mask = self._pad_tokens(
            encoder_hidden_states, num_text_tokens, "text context", "max_seq_len"
        )
        inputs = [
            hidden_states.to(dtype),
            encoder_hidden_states.to(dtype),
            encoder_attention_mask,
            self.time_proj(timestep.float() * 1000).float(),
        ]
        if target_timestep is not None:
            inputs.append(self.time_proj(target_timestep.float() * 1000).float())

        graph = self.graphs[graph_name(num_reference_tokens, num_text_tokens)]
        return self._prepare_output(graph(*(inputs + reference_inputs)), return_dict)


__all__ = [
    "RBLNFlux3Transformer2DModel",
]
