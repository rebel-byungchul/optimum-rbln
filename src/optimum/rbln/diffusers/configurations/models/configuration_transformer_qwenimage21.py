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


# Graph layouts of the transformer per profile (see `RBLNQwenImage21Transformer2DModelConfig`). The device DRAM of
# RBLN-CR13 is split into four chiplets and chiplet 0 receives most of the activation scratch, which adds up over the
# graphs and layers of a compiled model: the layers are therefore split into groups, and the buckets are few.
QWENIMAGE21_PROFILES = {
    # 1024-class targets (up to 4352 tokens) with up to 10 condition images of the 1024 class. Flash attention over four
    # 16384-token cache blocks (a chunk must stay inside one block): ten condition images with their text and the target
    # take about 54000 positions. The device keeps scratch of about a query chunk times the cache per layer and
    # executor, so the cache is not made larger than needed.
    "1024": {
        "attention": "flash",
        "cache_block_size": 16384,
        "cache_num_blocks": 4,
        "text_bucket": 512,
        "image_buckets": [4352],
        "target_buckets": [4352],
        "layer_groups": [(0, 16), (16, 32)],
        "verify_outputs": False,
    },
    # 2048-class targets (up to 16896 tokens) next to condition images of the 1024 class. Eager attention over one
    # 32256-token cache block.
    "2048": {
        "attention": "eager",
        "cache_block_size": 32256,
        "cache_num_blocks": 1,
        "text_bucket": 512,
        "image_buckets": [4352],
        "target_buckets": [16896],
        "layer_groups": [(0, 8), (8, 16), (16, 24), (24, 32)],
        # these graphs now and then give a different output for identical inputs on RBLN-CR13
        "verify_outputs": True,
    },
}


class RBLNQwenImage21Transformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Qwen-Image-2.1 Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options for the transformer of
    Qwen-Image-2.1 pipelines. The transformer runs over a key/value cache that stays on the device, as the original does with
    `kv_cache_mode="extract"`/`"cached"`: the text and condition-image tokens (the prefix) are written once per request
    in chunks, and later steps compute only the target image. Each layer group is one compiled graph with one executor
    per bucket: text with causal attention, condition images and the target with bidirectional attention (they share
    their executors); chunks are padded to their bucket.
    """

    def __init__(
        self,
        profile: str | None = None,
        attention: str | None = None,
        cache_block_size: int | None = None,
        cache_num_blocks: int | None = None,
        text_bucket: int | None = None,
        image_buckets: list[int] | None = None,
        target_buckets: list[int] | None = None,
        layer_groups: list[tuple[int, int]] | None = None,
        verify_outputs: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            profile (str | None): The image size class to compile for, which sets the defaults of the other options:
                "1024" (targets up to 4352 tokens, e.g. 1024x1024, with up to 10 condition images of the 1024 class)
                or "2048" (targets up to 16896 tokens, e.g. 2048x2048, with condition images of the 1024 class).
                Defaults to "1024".
            attention (str | None): "flash" (blocks of `cache_block_size` tokens; a chunk stays inside one block) or
                "eager" (one block). Defaults to the profile's.
            cache_block_size (int | None): Tokens per cache block. Defaults to the profile's.
            cache_num_blocks (int | None): Number of cache blocks. Defaults to the profile's.
            text_bucket (int | None): Text chunk length; longer text is written in several chunks (it is causal).
                Defaults to 512.
            image_buckets (list[int] | None): Condition-image chunk lengths in tokens. Defaults to the profile's.
            target_buckets (list[int] | None): Target-image chunk lengths in tokens. Defaults to the profile's.
            layer_groups (list[tuple[int, int]] | None): Layer ranges `[start, end)` compiled as separate graphs, in
                order. Defaults to the profile's.
            verify_outputs (bool | None): Run every layer-group call until two runs agree bit for bit. A workaround for
                graphs that now and then give a different output for identical inputs on RBLN-CR13 (seen with the
                2048-class graphs; it costs a second run of every call). Defaults to the profile's: True for "2048",
                False for "1024". Can be changed after loading with the model's `verify_outputs` attribute.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `profile` or `attention` is unknown.
        """
        super().__init__(**kwargs)
        self.profile = profile or "1024"
        if self.profile not in QWENIMAGE21_PROFILES:
            raise ValueError(f"profile must be one of {list(QWENIMAGE21_PROFILES)}, got {self.profile!r}")
        preset = QWENIMAGE21_PROFILES[self.profile]
        self.attention = attention or preset["attention"]
        if self.attention not in ("flash", "eager"):
            raise ValueError(f"attention must be 'flash' or 'eager', got {self.attention!r}")
        self.cache_block_size = cache_block_size or preset["cache_block_size"]
        self.cache_num_blocks = cache_num_blocks or preset["cache_num_blocks"]
        self.text_bucket = text_bucket or preset["text_bucket"]
        self.image_buckets = sorted(image_buckets or preset["image_buckets"])
        self.target_buckets = sorted(target_buckets or preset["target_buckets"])
        self.layer_groups = [tuple(group) for group in layer_groups or preset["layer_groups"]]
        self.verify_outputs = preset["verify_outputs"] if verify_outputs is None else bool(verify_outputs)

    @property
    def cache_capacity(self) -> int:
        return self.cache_block_size * self.cache_num_blocks


__all__ = [
    "RBLNQwenImage21Transformer2DModelConfig",
]
