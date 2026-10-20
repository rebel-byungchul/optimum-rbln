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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import rebel
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21TemporalTimesteps,
    QwenImage21Transformer2DModel,
)
from rebel.compile_context import CompileContext
from torch import nn

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNQwenImage21Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

MODULATION_GRAPH = "modulation"
# The attention of a layer-group call is told by the length of its `kind_tag` input (the values are not used): causal
# for text, bidirectional for condition images and the target, which share their executors (a condition image's
# output projection is computed and not used).
KIND_TAG = {"img": 1, "txt": 2}
# Rows of the original rotary table (QwenImage21Rope): positions 0..8191, then -1024..-1.
ROPE_TABLE_ROWS = 8192 + 1024


def layer_group_graph_name(start: int, end: int) -> str:
    return f"layers_{start}_{end}"


def rope_index_to_row(index: torch.Tensor) -> torch.Tensor:
    """Maps the original rotary positions (negative ones count from the end of the table) to table rows."""
    return torch.where(index < 0, index + ROPE_TABLE_ROWS, index)


class _RopeGather(nn.Module):
    """The original rotary table as real cos/sin, gathered by `[3, C]` row indices into `[C, head_dim]` (pairwise
    repeated, the layout of `_apply_rope`)."""

    def __init__(self, pos_embed: nn.Module):
        super().__init__()
        self.cos = nn.ModuleList(
            nn.Embedding.from_pretrained(f.real.float().repeat_interleave(2, dim=-1), freeze=True)
            for f in pos_embed.freqs
        )
        self.sin = nn.ModuleList(
            nn.Embedding.from_pretrained(f.imag.float().repeat_interleave(2, dim=-1), freeze=True)
            for f in pos_embed.freqs
        )

    def forward(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cat([table(index[axis]) for axis, table in enumerate(self.cos)], dim=-1)
        sin = torch.cat([table(index[axis]) for axis, table in enumerate(self.sin)], dim=-1)
        return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The original `apply_rotary_emb_qwen(use_real=False)` with real tables. x `[1, C, H, D]`, cos/sin `[C, D]`."""
    pairs = x.float().unflatten(-1, (-1, 2))
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return (x.float() * cos[None, :, None, :] + rotated * sin[None, :, None, :]).to(x.dtype)


class QwenImage21ModulationWrapper(nn.Module):
    """Timestep embedding → the modulation every layer uses and the output-norm scale, concatenated `[1, 5 * hidden]`.
    One small graph, so that the layer-group graphs take its output instead of each holding the timestep path."""

    def __init__(self, model: QwenImage21Transformer2DModel):
        super().__init__()
        self.timestep_embedder = model.time_text_embed.timestep_embedder
        self.modulation = model.modulation
        self.norm_out = model.norm_out

    def forward(self, timestep_proj: torch.Tensor) -> torch.Tensor:
        temb = self.timestep_embedder(timestep_proj)
        return torch.cat([self.modulation(temb), self.norm_out.linear(self.norm_out.silu(temb))], dim=-1)


class QwenImage21LayerGroupWrapper(nn.Module):
    """Layers `[start, end)` over the device KV cache, for one chunk of text, condition-image or target tokens.

    The joint sequence of the original is `[text | cond image 1 | text | ... | cond image n | text | target]` with
    block-causal attention: causal over text, bidirectional within an image. A chunk writes its keys and values at cache
    positions `cache_position` and attends over the cache (`mask` keeps the valid positions). The kind of the chunk is
    told by the length of `kind_tag`, so one compiled graph holds an executor per kind and bucket, sharing the weights.

    Inputs: x `[C, 4096]` (the first group: text `[C, 4096]` or latents `[C, 64]`), kind_tag, mod `[1, 5 * 4096]`,
    rope_index `[3, C]`, cache_position `[1, C]`, block_tables, mask `[1, capacity]`, then the cache of each layer.
    Output: the hidden state, or for the last group the noise prediction `[C, 64]` of a target chunk (`[1, 64]` for a
    prefix chunk, whose only effect is the cache write).
    """

    def __init__(
        self,
        model: QwenImage21Transformer2DModel,
        start: int,
        end: int,
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ):
        super().__init__()
        self.txt_in, self.img_in = model.txt_in, model.img_in
        self.norm_out, self.proj_out = model.norm_out, model.proj_out
        self.blocks = model.transformer_blocks[start:end]
        self.rope = _RopeGather(model.pos_embed)
        self.heads = model.config.num_attention_heads
        self.head_dim = model.config.attention_head_dim
        self.scale = nn.Parameter(torch.tensor(self.head_dim**-0.5), requires_grad=False)
        self.first, self.last = start == 0, end == model.config.num_layers
        self.flash = rbln_config.attention == "flash"
        self.block_size = rbln_config.cache_block_size

    def _attention(self, attn, x, cos, sin, kcache, vcache, cache_position, block_tables, mask, bidirectional):
        heads, head_dim, chunk = self.heads, self.head_dim, x.shape[1]
        q = attn.norm_q(attn.to_q(x).unflatten(-1, (heads, -1)))
        k = attn.norm_k(attn.to_k(x).unflatten(-1, (heads, -1)))
        v = attn.to_v(x).unflatten(-1, (heads, -1))
        q, k = _apply_rope(q.to(v.dtype), cos, sin), _apply_rope(k.to(v.dtype), cos, sin)  # [1, C, H, D]
        args = {
            "q": q.transpose(1, 2).reshape(1, heads, 1, chunk, head_dim),
            "k": k.transpose(1, 2).unsqueeze(2),
            "v": v.transpose(1, 2).unsqueeze(2),
            "kcache": kcache.unsqueeze(2),
            "vcache": vcache.unsqueeze(2),
            "scale": self.scale,
            "block_table": block_tables,
            "block_size": self.block_size,
            "is_bidirectional": bidirectional,
            "mask": mask,
        }
        if self.flash:
            out = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
                seq=cache_position[:, :1], partition=self.block_size, **args
            )
        else:
            out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(seq=cache_position[:, :1], **args)
        return attn.to_out[0](out.view(1, heads, chunk, head_dim).transpose(1, 2).flatten(2, 3))

    def forward(self, x, kind_tag, mod, rope_index, cache_position, block_tables, mask, *caches):
        kind = {tag: kind for kind, tag in KIND_TAG.items()}[kind_tag.shape[0]]
        mod = mod + kind_tag[:1, None] * 0.0  # (kind_tag must be used, or the compiler drops the input)
        if self.first:
            x = self.txt_in(x) if kind == "txt" else self.img_in(x)
        x = x[None]  # [1, C, hidden]
        hidden = x.shape[-1]
        shift1, gate1, shift2, gate2 = mod[:, None, : 4 * hidden].chunk(4, dim=-1)
        gate1, gate2 = gate1.tanh(), gate2.tanh()
        cos, sin = self.rope(rope_index)
        for i, block in enumerate(self.blocks):
            kcache, vcache = caches[2 * i], caches[2 * i + 1]
            h = block.img_norm1(x) * (1 + shift1)
            h = self._attention(
                block.attn, h, cos, sin, kcache, vcache, cache_position, block_tables, mask, kind != "txt"
            )
            x = x + gate1 * h
            x = x + gate2 * block.img_mlp(block.img_norm2(x) * (1 + shift2))
        if not self.last:
            return x[0]
        if kind == "img":
            return self.proj_out(self.norm_out.norm(x) * (1 + mod[:, None, 4 * hidden :]))[0]
        # A text chunk only writes the cache; its output still depends on the layers so that they are kept.
        return x[0, :1, :1].repeat(1, self.proj_out.out_features)


@dataclass
class _Segment:
    kind: str  # txt | img | tgt
    length: int
    rope_rows: torch.Tensor  # [3, length]


def build_segments(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]) -> list[_Segment]:
    """Splits the joint sequence into text, condition-image and target segments, with the rotary table rows the
    original `QwenImage21Rope` gives them. `image_pad_mask` `[S]` marks the image tokens (target included)."""
    is_image = image_pad_mask.tolist()
    segments, cursor, position = [], 0, 0
    for index, (_, height, width) in enumerate(img_shapes):
        block_start = is_image.index(True, cursor)
        text_len = block_start - cursor
        if text_len > 0:
            rows = torch.arange(position, position + text_len)
            segments.append(_Segment("txt", text_len, torch.stack([rows, rows, rows])))
        position += text_len
        frame = torch.full((height * width,), position)
        rows = torch.arange(-(height - height // 2), height // 2).repeat_interleave(width)
        cols = torch.arange(-(width - width // 2), width // 2).repeat(height)
        kind = "tgt" if index == len(img_shapes) - 1 else "img"
        segments.append(_Segment(kind, height * width, rope_index_to_row(torch.stack([frame, rows, cols]))))
        position += max(height, width)
        cursor = block_start + height * width
    if cursor < len(is_image):
        raise NotImplementedError("Text after the target image is not supported (the pipeline never builds it).")
    return segments


class _CacheLayout:
    """Cache positions of consecutive chunks. With flash attention a chunk must stay inside one block, so a chunk that
    would cross a block boundary starts at the next block. Raises ValueError when the cache is full."""

    def __init__(self, rbln_config: RBLNQwenImage21Transformer2DModelConfig, cursor: int = 0):
        self.flash = rbln_config.attention == "flash"
        self.block_size, self.capacity = rbln_config.cache_block_size, rbln_config.cache_capacity
        self.cursor = cursor

    def place(self, length: int) -> int:
        if self.flash:
            if length > self.block_size:
                raise ValueError(f"A chunk of {length} tokens does not fit in a cache block of {self.block_size}.")
            if self.cursor // self.block_size != (self.cursor + length - 1) // self.block_size:
                self.cursor = (self.cursor // self.block_size + 1) * self.block_size
        start = self.cursor
        self.cursor += length
        if self.cursor > self.capacity:
            raise ValueError(f"The KV cache of {self.capacity} tokens is full.")
        return start


class RBLNQwenImage21Transformer2DModel(RBLNModel):
    """
    RBLN implementation of QwenImage21Transformer2DModel for Qwen-Image-2.1 pipelines.

    The text and condition-image keys and values are step-independent (`causal_condition`), so the original prefills
    them at the first step (`kv_cache_mode="extract"`) and later steps compute only the target image
    (`kv_cache_mode="cached"`). The same happens here over a key/value cache that stays on the RBLN devices: the prefix
    is written in chunks (text in pieces of `text_bucket` tokens, each condition image as one chunk) and the target is
    computed after it. The CFG contexts (`cache_context("cond"/"uncond")`) keep their own prefix regions.

    The layers run as a few layer-group graphs (see `layer_groups`), each with one executor per attention kind and bucket;
    the timestep embedding and modulation are one more graph. With a list of devices, the layer groups are spread over
    them in order. Only the sinusoidal timestep projection is computed on the host, in float32.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImage21Transformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        # (no runtimes when loaded with `create_runtimes=False`)
        names = [cfg.compiled_model_name for cfg in self.rbln_config.compile_cfgs]
        self.graphs = dict(zip(names, self.model, strict=True)) if self.model else {}
        self.verify_outputs = self.rbln_config.verify_outputs
        self._context = "cond"
        self._request = None
        self._modulations = {}

    @property
    def transformer_blocks(self) -> range:
        # The pipeline only counts the blocks (to size its own cache object, which is not used here).
        return range(self.config.num_layers)

    @contextmanager
    def cache_context(self, name: str):
        """Marks the CFG context of the calls inside (as the diffusers `CacheMixin`): each keeps its own prefix."""
        previous, self._context = self._context, name
        try:
            yield
        finally:
            self._context = previous

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config sets the profile it shares with the VAE.
        return rbln_config

    @classmethod
    def _graph_devices(cls, rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> dict[str, int]:
        """The device of each graph: with a list of devices, the layer groups are spread over them in order (the
        modulation graph goes with the first); otherwise `device_map`."""
        devices = rbln_config.device
        if not isinstance(devices, (list, tuple)):
            return rbln_config.device_map
        groups = rbln_config.layer_groups
        placement = {MODULATION_GRAPH: devices[0]}
        for i, (start, end) in enumerate(groups):
            placement[layer_group_graph_name(start, end)] = devices[i * len(devices) // len(groups)]
        return placement

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        devices = cls._graph_devices(rbln_config)
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            name = compile_config.compiled_model_name
            context = CompileContext(use_weight_sharing=True)
            caches = {}
            if name == MODULATION_GRAPH:
                wrapped = QwenImage21ModulationWrapper(model).eval()
            else:
                start, end = (int(n) for n in name.split("_")[1:])
                wrapped = QwenImage21LayerGroupWrapper(model, start, end, rbln_config).eval()
                # The cache of each layer stays on the device between calls: its example input is the marked tensor.
                for layer in range(start, end):
                    for kind in ("k", "v"):
                        caches[f"{kind}cache_{layer}"] = cls._cache_tensor(rbln_config)
                        context.mark_static_address(caches[f"{kind}cache_{layer}"], f"{kind}cache_{layer}")
            compiled_models[name] = cls.compile(
                wrapped,
                rbln_compile_config=compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=devices[name],
                compile_context=context,
                example_inputs=cls._example_inputs(compile_config, caches),
            )
        return compiled_models

    @staticmethod
    def _example_inputs(
        compile_config: RBLNCompileConfig, caches: dict[str, torch.Tensor]
    ) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        # The marked caches, valid indices for the integer inputs (rotary rows, cache positions, block tables), zeros
        # otherwise.
        def example(name, shape, dtype):
            if name in caches:
                return caches[name]
            if name in ("cache_position", "block_tables"):
                return torch.arange(shape[-1], dtype=getattr(torch, dtype)).reshape(shape)
            return torch.zeros(shape, dtype=getattr(torch, dtype))

        multiple = compile_config.is_multiple_input_info
        input_infos = compile_config.input_info if multiple else [compile_config.input_info]
        examples = [[example(*info) for info in input_info] for input_info in input_infos]
        return examples if multiple else examples[0]

    @staticmethod
    def _cache_tensor(rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> torch.Tensor:
        return torch.zeros(rbln_config.cache_num_blocks, 32, rbln_config.cache_block_size, 128)

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        devices = cls._graph_devices(rbln_config)
        if any(name not in devices for name in names):
            cls._raise_missing_compiled_file_error(names)
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=devices[name],
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
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ) -> RBLNQwenImage21Transformer2DModelConfig:
        num_layers, hidden = (
            model_config.num_layers,
            model_config.num_attention_heads * model_config.attention_head_dim,
        )
        if rbln_config.layer_groups[0][0] != 0 or rbln_config.layer_groups[-1][1] != num_layers:
            raise ValueError(
                f"layer_groups must cover the {num_layers} layers in order, got {rbln_config.layer_groups}"
            )
        cache_shape = list(cls._cache_tensor(rbln_config).shape)
        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=MODULATION_GRAPH, input_info=[("timestep_proj", [1, 256], "float32")]
            )
        ]
        # The executors of a graph share its weights, laid out for the executor compiled first: the largest
        # bidirectional bucket (the target, run on every step) goes first, the text last.
        buckets = sorted(set(rbln_config.image_buckets + rbln_config.target_buckets), reverse=True)
        executors = [("img", n) for n in buckets] + [("txt", rbln_config.text_bucket)]
        for start, end in rbln_config.layer_groups:
            input_info = []
            for kind, chunk in executors:
                width = (hidden if kind == "txt" else model_config.in_channels) if start == 0 else hidden
                info = [
                    ("x", [chunk, width], "float32"),
                    ("kind_tag", [KIND_TAG[kind]], "float32"),
                    ("mod", [1, 5 * hidden], "float32"),
                    ("rope_index", [3, chunk], "int64"),
                    ("cache_position", [1, chunk], "int32"),
                    ("block_tables", [rbln_config.cache_num_blocks], "int16"),
                    ("mask", [1, rbln_config.cache_capacity], "float32"),
                ]
                for layer in range(start, end):
                    info += [(f"{kind_}cache_{layer}", cache_shape, "float32") for kind_ in ("k", "v")]
                input_info.append(info)
            compile_cfgs.append(
                RBLNCompileConfig(compiled_model_name=layer_group_graph_name(start, end), input_info=input_info)
            )
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    # ------------------------------------------------------------------------------------------------ inference
    def _run_graph(self, graph, *args) -> torch.Tensor:
        output = graph(*args)
        if not self.verify_outputs:
            return output
        for _ in range(6):  # until two runs agree
            again = graph(*args)
            if torch.equal(again, output):
                return output
            output = again
        raise RuntimeError("A transformer graph gave a different output on every run.")

    def _modulation(self, timestep: float) -> torch.Tensor:
        """The `mod` input of the layer groups for a timestep (the prefix uses 0, like the original)."""
        if timestep not in self._modulations:
            if len(self._modulations) > 64:
                self._modulations.clear()
            proj = self.time_proj(torch.tensor([timestep], dtype=torch.float32)).float()
            self._modulations[timestep] = self._run_graph(self.graphs[MODULATION_GRAPH], proj)
        return self._modulations[timestep]

    def _run_chunk(self, kind, x, timestep, rope, position, mask):
        """Runs a chunk through the layer groups; `timestep` is 0 for the prefix, like the original."""
        args = (
            torch.zeros(KIND_TAG["txt" if kind == "txt" else "img"]),
            self._modulation(timestep),
            rope,
            position,
            torch.arange(self.rbln_config.cache_num_blocks, dtype=torch.int16),
            mask,
        )
        for start, end in self.rbln_config.layer_groups:
            x = self._run_graph(self.graphs[layer_group_graph_name(start, end)], x, *args)
        return x

    def _bucket(self, kind: str, length: int) -> int:
        buckets = {
            "txt": [self.rbln_config.text_bucket],
            "img": self.rbln_config.image_buckets,
            "tgt": self.rbln_config.target_buckets,
        }[kind]
        for bucket in buckets:
            if bucket >= length:
                return bucket
        option = {"img": "image_buckets", "tgt": "target_buckets"}.get(kind, "text_bucket")
        raise ValueError(
            f"{length} {'condition-image' if kind == 'img' else 'target'} tokens do not fit the compiled buckets "
            f"(`{option}={buckets}`, profile {self.rbln_config.profile!r})."
        )

    def _prefix_chunks(self, segments):
        """(kind, bucket, segment, offset, length) of the prefix chunks in order: text in pieces of `text_bucket`
        tokens (text is causal, so this gives the same result), every condition image whole."""
        chunks = []
        for segment in segments:
            if segment.kind == "txt":
                for offset in range(0, segment.length, self.rbln_config.text_bucket):
                    length = min(self.rbln_config.text_bucket, segment.length - offset)
                    chunks.append(("txt", self.rbln_config.text_bucket, segment, offset, length))
            elif segment.kind == "img":
                chunks.append(("img", self._bucket("img", segment.length), segment, 0, segment.length))
        return chunks

    def _prefix_end(self, segments, start: int, target_bucket: int) -> int | None:
        """Where the prefix ends when written from `start`, or None if it and the target do not fit."""
        layout = _CacheLayout(self.rbln_config, start)
        try:
            for _, bucket, *_ in self._prefix_chunks(segments):
                layout.place(bucket)
            end = layout.cursor
            layout.place(target_bucket)
        except ValueError:
            return None
        return end

    def _write_prefix(self, segments, text, text_valid, cond_latents, start: int) -> tuple[torch.Tensor, int]:
        """Writes the prefix chunks to the cache from `start`; returns the valid-position mask and the prefix end.
        Each chunk sees the mask of the chunks up to itself (later positions may still hold another request's keys)."""
        layout = _CacheLayout(self.rbln_config, start)
        mask = torch.zeros(1, self.rbln_config.cache_capacity)
        text_ptr = image_ptr = 0
        for kind, bucket, segment, offset, length in self._prefix_chunks(segments):
            if kind == "txt":
                rows = text[text_ptr + offset : text_ptr + offset + length]
                valid = text_valid[text_ptr + offset : text_ptr + offset + length].float()
                if offset + length == segment.length:
                    text_ptr += segment.length
            else:
                rows, valid = cond_latents[image_ptr : image_ptr + length], torch.ones(length)
                image_ptr += length
            position = layout.place(bucket)
            mask[0, position : position + length] = valid
            x, rope = self._pad_chunk(rows, segment.rope_rows[:, offset : offset + length], bucket)
            self._run_chunk(kind, x, 0.0, rope, self._positions(position, bucket), mask.clone())
        return mask, layout.cursor

    @staticmethod
    def _pad_chunk(rows: torch.Tensor, rope_rows: torch.Tensor, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(bucket, rows.shape[-1])
        x[: rows.shape[0]] = rows.float()
        rope = torch.zeros(3, bucket, dtype=torch.long)
        rope[:, : rows.shape[0]] = rope_rows
        return x, rope

    @staticmethod
    def _positions(start: int, bucket: int) -> torch.Tensor:
        return (start + torch.arange(bucket, dtype=torch.int32))[None]

    def _forward_one(self, key, hidden_states, timestep, encoder_hidden_states, text_valid, slots, segments, mode):
        """One sample: write (or reuse) its prefix, then compute the target."""
        target = segments[-1]
        num_target = target.length
        num_cond = hidden_states.shape[0] - num_target
        text = encoder_hidden_states[~slots]  # the image-slot rows are replaced by latents, like in the original
        text_valid = text_valid[~slots]
        target_bucket = self._bucket("tgt", num_target)

        request = self._request
        context = request["contexts"].get(key)
        if context is None:
            context = {"volatile": True, "start": 0}
            end = None if request["all_volatile"] else self._prefix_end(segments, request["cursor"], target_bucket)
            if end is not None:  # a prefix region kept for the whole request
                context = {"volatile": False, "start": request["cursor"]}
                request["cursor"] = end
            else:  # out of room: from now on every context rewrites its prefix from 0 on every call
                request["all_volatile"] = True
            request["contexts"][key] = context
        if request["all_volatile"]:
            context.update(volatile=True, start=0)
        if context.get("mask") is None or mode is None or context["volatile"]:
            if self._prefix_end(segments, context["start"], target_bucket) is None:
                raise ValueError(
                    f"The prompt, condition images and target ({num_target} tokens) do not fit the KV cache of "
                    f"{self.rbln_config.cache_capacity} tokens (profile {self.rbln_config.profile!r})."
                )
            context["mask"], context["end"] = self._write_prefix(
                segments, text, text_valid, hidden_states[:num_cond], context["start"]
            )

        # The target goes right after the prefixes (attention costs grow with the position); its keys and values are
        # rewritten on every call.
        prefix_end = context["end"] if request["all_volatile"] else request["cursor"]
        position = _CacheLayout(self.rbln_config, prefix_end).place(target_bucket)
        mask = context["mask"].clone()
        mask[0, position : position + num_target] = 1
        x, rope = self._pad_chunk(hidden_states[num_cond:], target.rope_rows, target_bucket)
        out = self._run_chunk("tgt", x, float(timestep), rope, self._positions(position, target_bucket), mask)
        return out[:num_target]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized QwenImage21Transformer2DModel.

        Args:
            hidden_states (torch.Tensor): Packed latents of shape `(batch_size, image_seq_len, in_channels)`,
                condition images first and the target image last.
            encoder_hidden_states (torch.Tensor): Prompt embeddings of shape `(batch_size, text_seq_len, 4096)`, with
                a row per image slot.
            timestep (torch.Tensor): Timesteps in `[0, 1]` of shape `(batch_size,)`.
            img_shapes (list): `(frames, height, width)` of every condition image and the target, per sample. The
                layout of the first sample is used, like the original.
            img_mask (torch.Tensor): Marks the image slots among the prompt tokens (each expands to 4 tokens).
            encoder_hidden_states_mask (torch.Tensor | None): 1 for the prompt tokens, 0 for batch padding.
            attention_kwargs (dict | None): Unused; accepted for compatibility with the diffusers signature.
            kv_cache (Any | None): The pipeline's cache object, unused: the cache is on the device.
            kv_cache_mode (str | None): "extract" writes the prefix (first step), "cached" reuses it, None rewrites it
                on every call.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_output.Transformer2DModelOutput` | tuple)
        """
        repeats = torch.where(img_mask, 4, 1)[0]
        pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        if int(pad_mask.sum()) != hidden_states.shape[1]:
            raise RuntimeError(
                f"shape mismatch: {hidden_states.shape[1]} image tokens for {int(pad_mask.sum())} image slots"
            )
        segments = build_segments(pad_mask, img_shapes[0])
        if self._context == "cond" and kv_cache_mode in (None, "extract"):  # a request starts with its cond call
            self._request = {"cursor": 0, "all_volatile": False, "contexts": {}}
        if self._request is None:
            raise RuntimeError(
                "The first call of a request must be in the 'cond' context with kv_cache_mode 'extract'."
            )

        slots = img_mask[0][: encoder_hidden_states.shape[1]]
        timesteps = torch.as_tensor(timestep).reshape(-1)
        outputs = []
        for b in range(hidden_states.shape[0]):
            text_valid = (
                torch.ones(encoder_hidden_states.shape[1], dtype=torch.bool)
                if encoder_hidden_states_mask is None
                else encoder_hidden_states_mask[b].bool()
            )
            outputs.append(
                self._forward_one(
                    (self._context, b),
                    hidden_states[b],
                    timesteps[b if timesteps.numel() > 1 else 0],
                    encoder_hidden_states[b],
                    text_valid,
                    slots,
                    segments,
                    kv_cache_mode,
                )
            )
        return self._prepare_output(torch.stack(outputs).float(), return_dict)


__all__ = [
    "RBLNQwenImage21Transformer2DModel",
]
