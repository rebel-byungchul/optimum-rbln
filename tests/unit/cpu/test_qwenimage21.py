"""CPU checks for the Qwen-Image 2.1 RBLN port: the host-side metadata and the rewritten modules must match diffusers.

Skipped unless diffusers ships `QwenImage21Pipeline` (diffusers>=0.41).
"""

import pytest
import torch


diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip("diffusers without Qwen-Image 2.1", allow_module_level=True)

from diffusers import AutoencoderKLQwenImage21  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage21 import (  # noqa: E402
    QwenImage21Rope,
    QwenImage21TemporalTimesteps,
)

from optimum.rbln.diffusers.pipelines.qwenimage21 import vae as V  # noqa: E402
from optimum.rbln.diffusers.pipelines.qwenimage21.dit_kv import (  # noqa: E402
    CacheLayout,
    KVConfig,
    RopeGather,
    build_segments,
)
from optimum.rbln.diffusers.pipelines.qwenimage21.ops import ExactTimestepProj  # noqa: E402


torch.set_grad_enabled(False)


def _layout(text_lens, cond_grids, tgt_grid):
    mask = []
    for i, (h, w) in enumerate(cond_grids):
        mask += [False] * text_lens[i] + [True] * (h * w // 4)
    mask += [False] * text_lens[len(cond_grids)] + [True] * (tgt_grid[0] * tgt_grid[1] // 4)
    vlm = torch.tensor(mask)
    pad = torch.repeat_interleave(vlm, torch.where(vlm, 4, 1))
    shapes = [(1, h, w) for h, w in cond_grids] + [(1, *tgt_grid)]
    return pad, shapes


@pytest.mark.parametrize(
    "text_lens,cond_grids,tgt_grid",
    [([51], [], (64, 64)), ([20, 5, 40], [(64, 64), (56, 72)], (48, 86)), ([7, 30], [(12, 10)], (158, 106))],
)
def test_segments_and_rope_gather_match_original_rope(text_lens, cond_grids, tgt_grid):
    pad, shapes = _layout(text_lens, cond_grids, tgt_grid)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    ref = rope(shapes, pad, torch.device("cpu"))  # [S, 64] complex
    segs = build_segments(pad, shapes)
    assert sum(s.length for s in segs) == pad.numel()
    assert [s.kind for s in segs][-1] == "tgt"
    assert sum(s.kind == "img" for s in segs) == len(cond_grids)
    gather = RopeGather(rope)
    rows = torch.cat([s.rope_rows for s in segs], dim=1)
    cos, sin = gather(rows)
    torch.testing.assert_close(cos[:, ::2], ref.real.float(), rtol=0, atol=0)
    torch.testing.assert_close(sin[:, ::2], ref.imag.float(), rtol=0, atol=0)


def test_exact_timestep_proj_is_bit_exact():
    g = torch.Generator().manual_seed(0)
    t = torch.cat([torch.tensor([1.0, 0.978453, 0.414568, 0.0]), torch.rand(60, generator=g) ** 3]).to(torch.bfloat16)
    ref = QwenImage21TemporalTimesteps(256)(t).to(torch.bfloat16)
    out = ExactTimestepProj()(t.float())
    assert torch.equal(out, ref)


def test_cache_layout_keeps_flash_chunks_inside_one_partition():
    lay = CacheLayout(KVConfig(part=1024, n_part=8))
    starts = [(lay.place(n), n) for n in (100, 1000, 300, 900, 512)]
    for s, n in starts:
        assert s // 1024 == (s + n - 1) // 1024
    with pytest.raises(ValueError):
        lay.place(1025)


@pytest.fixture(scope="module")
def tiny_vae():
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage21(
        base_dim=16,
        decoder_base_dim=24,
        z_dim=8,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        in_channels=4,
        out_channels=4,
        is_residual=True,
    )
    for p in vae.parameters():  # random but well-scaled weights
        p.data.normal_(0, 0.05)
    return vae.eval()


def test_vae_2d_decoder_and_chain_match_original(tiny_vae):
    z = torch.randn(1, 8, 1, 12, 10)
    ref = tiny_vae.decode(z, return_dict=False)[0][:, :, 0]
    full = V.QwenImage21VAEDecoder2D(tiny_vae).eval()
    torch.testing.assert_close(full(z[:, :, 0]), ref, rtol=1e-4, atol=1e-4)


def test_vae_2d_encoder_matches_original(tiny_vae):
    x = torch.rand(1, 4, 1, 64, 96) * 2 - 1
    ref = tiny_vae.encode(x).latent_dist.mode()[:, :, 0]
    out = V.QwenImage21VAEEncoder2D(tiny_vae).eval()(x[:, :, 0])
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_dit_layer_groups_are_balanced_over_devices():
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_dit import UNIFIED, RBLNDiTKV

    for _, groups in UNIFIED.values():
        assert groups[0][0] == 0 and groups[-1][1] == 32
        assert all(a[1] == b[0] for a, b in zip(groups, groups[1:], strict=False))
    rt = RBLNDiTKV.__new__(RBLNDiTKV)
    rt.devices, rt.reserve = [0, 1], {0: 10.7, 1: 10.0}
    rt.models = {"eager": [(0, 8, "a", []), (8, 16, "b", []), (16, 24, "c", []), (24, 32, "d", [])]}
    rt.sizes = {"a": 11.1, "b": 11.2, "c": 11.2, "d": 11.0}  # chiplet-0 GiB
    assert rt._placement("eager") == [("a", 0), ("b", 0), ("c", 1), ("d", 1)]
    rt.reserve = {0: 25.0}
    assert [d for _, d in rt._placement("eager")] == [0, 1, 1, 1]
    rt.devices, rt.reserve = [3], {}
    assert [d for _, d in rt._placement("eager")] == [3, 3, 3, 3]
