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


class RBLNFlux3Transformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN FLUX.3 Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for the transformer of FLUX.3 pipelines such as FLUX 3 Image Klein.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: int | tuple[int, int] | None = None,
        max_seq_len: int | None = None,
        text_token_buckets: list[int] | None = None,
        image_rope_scale: float | None = None,
        reference_token_buckets: list[int] | None = None,
        share_weights: bool | None = None,
        attention_query_chunk_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            sample_size (int | tuple[int, int] | None): The spatial size of the packed latents as
                (height, width), i.e. the image size divided by 16. A single integer is used for both
                sides. Defaults to 64 (1024 x 1024 images).
            max_seq_len (int | None): The longest text context the transformer is compiled for. A shorter context
                is padded to the smallest text bucket that holds it and the padding is masked out of attention, with
                the same result. Defaults to 1040.
            text_token_buckets (list[int] | None): The text lengths to compile graphs for, `max_seq_len` among them;
                a step at a shorter length is faster (on RBLN-CR13 at 1024x1024, text-to-image: 151 ms at 320 tokens,
                198 ms at 1040). Every graph gets them except the image-conditioned graph of the
                largest reference capacity (when there are several), which keeps `max_seq_len` only since its working
                memory is the largest. Defaults to `max_seq_len` alone.
            image_rope_scale (float | None): The scale of the image grid positions in the rotary embeddings, as used
                by the pipeline that builds `img_ids`. The rotary embeddings are compiled into the graph. Defaults to
                the scale of the diffusers FLUX.3 latents step.
            reference_token_buckets (list[int] | None): Reference-token capacities to compile image-conditioned graphs
                for, one graph per capacity next to the text-to-image graph. A call with reference images runs the
                smallest graph that holds their tokens (the others are masked out). Defaults to no image-conditioned
                graph.
            share_weights (bool | None): Whether the graphs share the transformer weights on the device. Sharing
                saves a copy of the weights per image-conditioned graph (about 18 GiB with the two default graphs
                of FLUX 3 Image Klein on RBLN-CR13). The image-conditioned graphs are compiled first, so that the
                shared weights are laid out for them; on RBLN-CR13 every graph then runs as fast as without sharing
                (one step at 1040 text tokens: 198 ms text-to-image, 418 ms with 4096 reference tokens, 929 ms with
                10240). Defaults to True.
            attention_query_chunk_size (int | None): Attention runs in chunks of this many queries, so a graph holds
                the attention scores of one chunk at a time rather than of the whole sequence; 0 attends all queries at
                once. On RBLN-CR13 this takes the transformer of FLUX 3 Image Klein with the two default image-conditioned
                graphs from 50.5 to 30.2 GiB at the same step time. Defaults to 1024.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.sample_size = sample_size
        if isinstance(self.sample_size, int):
            self.sample_size = (self.sample_size, self.sample_size)
        self.max_seq_len = max_seq_len
        self.text_token_buckets = sorted(text_token_buckets) if text_token_buckets else None
        self.image_rope_scale = image_rope_scale
        self.reference_token_buckets = sorted(reference_token_buckets or [])
        self.share_weights = True if share_weights is None else share_weights
        self.attention_query_chunk_size = attention_query_chunk_size


__all__ = [
    "RBLNFlux3Transformer2DModelConfig",
]
