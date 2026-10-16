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
        image_rope_scale: float | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            sample_size (int | tuple[int, int] | None): The spatial size of the packed latents as
                (height, width), i.e. the image size divided by 16. A single integer is used for both
                sides. Defaults to 64 (1024 x 1024 images).
            max_seq_len (int | None): The length of the text context the transformer is compiled for.
                The pipeline pads the prompt to a multiple of 80 tokens; the tokens between that window and
                `max_seq_len` are masked out of attention, so any prompt up to this length runs on one graph.
                Rounded up to a multiple of 80. Defaults to 1040.
            image_rope_scale (float | None): The scale of the image grid positions in the rotary embeddings, as used
                by the pipeline that builds `img_ids`. The rotary embeddings are compiled into the graph.
                Defaults to 0.5.
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
        self.image_rope_scale = image_rope_scale


__all__ = [
    "RBLNFlux3Transformer2DModelConfig",
]
