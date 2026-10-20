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


class RBLNAutoencoderKLQwenImage21Config(RBLNModelConfig):
    """
    Configuration class for the RBLN Qwen-Image-2.1 Variational Autoencoder.

    This class inherits from RBLNModelConfig and provides specific configuration options for the VAE of Qwen-Image-2.1
    pipelines. The decoder is compiled for every image size in `sample_sizes` and the encoder for every condition-image size in
    `encoder_sample_sizes`, each as a chain of graphs (see `RBLNAutoencoderKLQwenImage21`).
    """

    def __init__(
        self,
        sample_sizes: list[tuple[int, int]] | None = None,
        encoder_sample_sizes: list[tuple[int, int]] | None = None,
        decoder_row_tiles: int | None = None,
        encoder_row_tiles: int | None = None,
        verify_outputs: bool | None = None,
        row_tiles: dict[str, int] | None = None,
        max_chain_gib: float | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            sample_sizes (list[tuple[int, int]] | None): `(height, width)` in pixels of the images to decode, multiples
                of 16. Defaults to `[(1024, 1024)]`.
            encoder_sample_sizes (list[tuple[int, int]] | None): `(height, width)` in pixels of the condition images
                to encode, multiples of 16. Defaults to none (no encoder).
            decoder_row_tiles (int | None): The decoder runs the part after its mid block in this many row tiles
                (exact: each tile has a halo of extra rows), which bounds the device memory of a large size. Defaults
                to 1 up to the 1024 class and 8 above.
            encoder_row_tiles (int | None): The same for the encoder part before its mid block. Defaults to 1.
            verify_outputs (bool | None): Run every graph call until two runs agree bit for bit. A workaround for
                graphs that now and then give a different output for identical inputs on RBLN-CR13. Defaults to False.
            row_tiles (dict[str, int] | None): Row tiles of single chains, by `"{decoder|encoder}_{height}x{width}"`.
                Set at compile time for the sizes whose chain does not compile in one piece (shape-specific compiler
                errors); takes precedence over the defaults.
            max_chain_gib (float | None): Chiplet-0 GiB a tiled chain may take on the device (a chain of a size is
                loaded whole); a larger one is compiled in more row tiles. Defaults to 25, which fits next to the
                2048-class transformer layers on an RBLN-CR13 device (with the encoder on another device).
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If a size is not a multiple of 16.
        """
        super().__init__(**kwargs)
        self.sample_sizes = [tuple(size) for size in sample_sizes or [(1024, 1024)]]
        self.encoder_sample_sizes = [tuple(size) for size in encoder_sample_sizes or []]
        for height, width in self.sample_sizes + self.encoder_sample_sizes:
            if height % 16 or width % 16:
                raise ValueError(f"VAE sizes must be multiples of 16, got {height}x{width}")
        self.decoder_row_tiles = decoder_row_tiles
        self.encoder_row_tiles = encoder_row_tiles or 1
        self.verify_outputs = bool(verify_outputs)
        self.row_tiles = dict(row_tiles or {})
        self.max_chain_gib = max_chain_gib or 25.0

    def tiles(self, kind: str, height: int, width: int) -> int:
        """Row tiles of a chain: `row_tiles`, else `encoder_row_tiles` for the encoder, and `decoder_row_tiles` or 1
        up to the 1024 class and 8 above for the decoder."""
        if f"{kind}_{height}x{width}" in self.row_tiles:
            return self.row_tiles[f"{kind}_{height}x{width}"]
        if kind == "encoder":
            return self.encoder_row_tiles
        if self.decoder_row_tiles is not None:
            return self.decoder_row_tiles
        return 1 if height * width <= 1088 * 1088 else 8


__all__ = [
    "RBLNAutoencoderKLQwenImage21Config",
]
