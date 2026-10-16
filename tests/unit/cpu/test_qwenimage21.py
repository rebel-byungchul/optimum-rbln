"""CPU checks for the Qwen-Image 2.1 RBLN port: the host-side metadata and the rewritten modules must match diffusers.

Skipped unless diffusers ships `QwenImage21Pipeline` (diffusers>=0.41).
"""

import math
import os

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
from optimum.rbln.diffusers.pipelines.qwenimage21.ops import ExactTimestepProj, split_timestep  # noqa: E402


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


def test_exact_timestep_proj_matches_fp32_original():
    g = torch.Generator().manual_seed(0)
    t = torch.cat([torch.tensor([1.0, 0.9785, 0.84512, 0.4146, 1e-5, 0.0]), torch.rand(200, generator=g) ** 3])
    pieces = split_timestep(t)
    # three non-negative bf16 values that add up to t exactly (the table lookup needs both)
    assert (pieces >= 0).all() and torch.equal(pieces.to(torch.bfloat16).float(), pieces)
    assert torch.equal(pieces.sum(-1), t)
    # exact sinusoid (fp64): the lookup drops pieces below 2^-23 (phase ≤ 1.2e-4 rad); the fp32 original itself rounds
    # phases of up to 1000 rad by about as much
    freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).double()

    def exact(x):
        a = 1000 * x.double()[:, None] * freqs[None]
        return torch.cat([a.cos(), a.sin()], -1)

    out = ExactTimestepProj()(pieces)
    torch.testing.assert_close(out.double(), exact(t), rtol=0, atol=1.3e-4)
    torch.testing.assert_close(out, QwenImage21TemporalTimesteps(256)(t), rtol=0, atol=2e-4)
    # a bf16 timestep alone is looked up directly
    tb = t.to(torch.bfloat16).float()
    torch.testing.assert_close(ExactTimestepProj()(tb).double(), exact(tb), rtol=0, atol=1.3e-4)


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
    vae = AutoencoderKLQwenImage21(  # the checkpoint's structure (dim_mult, res blocks, downsampling) with fewer channels
        base_dim=16,
        decoder_base_dim=24,
        z_dim=8,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        temperal_downsample=[False, True, True, True],
        in_channels=4,
        out_channels=4,
        is_residual=True,
    )
    for p in vae.parameters():  # random but well-scaled weights
        p.data.normal_(0, 0.05)
    return vae.eval()


def test_vae_2d_decoder_matches_original(tiny_vae):
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


def test_vae_row_tiles_match_untiled(tiny_vae):
    dec = V.QwenImage21VAEDecoder2D(tiny_vae).eval()
    gd = {n: F(dec).eval() for n, F, _, _ in V.DECODER_PROGRAM}
    z = torch.randn(1, 8, 40, 6)
    full = V.run_program(V.DECODER_PROGRAM, gd, {"z": z})["out"]
    torch.testing.assert_close(full, dec(z), rtol=1e-4, atol=1e-4)
    tiled = V.run_program(V.DECODER_PROGRAM, gd, {"z": z}, n_tiles=4)["out"]
    torch.testing.assert_close(tiled, full, rtol=1e-4, atol=1e-4)

    enc = V.QwenImage21VAEEncoder2D(tiny_vae).eval()
    ge = {n: F(enc).eval() for n, F, _, _ in V.ENCODER_PROGRAM}
    x = torch.rand(1, 4, 192, 48) * 2 - 1
    kw = {"tiled": V.ENC_TILED_STEPS, "halo": V.ENC_TILE_HALO, "base": "x", "unit": V.ENC_TILE_UNIT}
    full = V.run_program(V.ENCODER_PROGRAM, ge, {"x": x})["z"]
    torch.testing.assert_close(full, enc(x), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        V.run_program(V.ENCODER_PROGRAM, ge, {"x": x}, n_tiles=3, **kw)["z"], full, rtol=1e-4, atol=1e-4
    )


def _fake_dit(xxl=False):
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_dit import RBLNDiTKV

    rt = RBLNDiTKV.__new__(RBLNDiTKV)
    rt.prompt_lens, rt._bucket_cache = [], {}
    rt.models = {
        "flash": [(0, 32, "f", [("txt", 512), ("img", 4352), ("tgt", 4352)])],
        "eager": [(0, 32, "e", [("txt", 512), ("img", 4352), ("tgt", 16896)])],
        "xl": [(0, 32, "x", [("tgt", 16320)])],
    }
    rt.prefix_models = {
        "xl": [(0, 0, 32, "xp", [("txt", 512), ("img", 4352)])],
        "eager": [(0, 0, 32, "ep", [("txt", 512), ("img", 4352), ("img", 8448), ("img", 12288), ("img", 14336)])],
    }
    if xxl:  # the xxl bank (dit_paged.py): per-layer weight graphs + shared attention over a block cache
        from optimum.rbln.diffusers.pipelines.qwenimage21.dit_paged import BLOCKS_PER_LAYER, PART, PIECE
        from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_paged import PagedDiT

        rec = {"l0": 0, "l1": 32, "n_layers": 32, "blocks_per_layer": BLOCKS_PER_LAYER, "part": PART, "piece": PIECE}
        rt.paged = PagedDiT([(rec, "")], [0], None)
        rt.models["xxl"] = [(0, 32, None, [])]
    return rt


@pytest.mark.parametrize(
    "text_lens,cond_grids,tgt_grid,expected",
    [
        ([51], [], (64, 64), "flash"),  # 1024² text-to-image
        ([12] + [5] * 9 + [40], [(64, 64)] * 10, (64, 64), "flash"),  # 1024², 10 references
        ([60], [], (128, 128), "eager"),  # 2048²
        ([30, 40], [(96, 96)], (128, 128), "eager"),  # 2048² with a 9216-token condition (output_resolution 1536)
        ([20, 5, 5, 5, 40], [(64, 64)] * 4, (158, 102), "xl"),  # 2528x1632 with 4 references
        ([20, 5, 5, 5, 40], [(64, 64)] * 4, (128, 128), "xxl"),  # 2048² with 4 references
        ([20] + [5] * 5 + [40], [(64, 64)] * 6, (128, 128), "xxl"),  # 2048² with 6 references
        ([30, 40], [(128, 128)], (128, 128), "xxl"),  # 2048² with an output_resolution-2048 condition image
        ([30, 5, 40], [(128, 128)] * 2, (128, 128), "xxl"),  # ... with two of them
        ([30, 5, 5, 5, 40], [(128, 128)] * 4, (128, 128), "xxl"),  # ... with four of them
        ([30] + [5] * 8 + [40], [(128, 128)] * 9, (128, 128), "xxl"),  # ... with nine of them
        ([30] + [5] * 9 + [40], [(128, 128)] * 10, (128, 128), None),  # ten: beyond the block cache
    ],
)
def test_dit_cfg_selection(text_lens, cond_grids, tgt_grid, expected):
    rt = _fake_dit(xxl=True)
    pad, shapes = _layout(text_lens, cond_grids, tgt_grid)
    segs = build_segments(pad, shapes)
    if expected is None:
        with pytest.raises(ValueError):
            rt._cfg_for(segs[-1].length, segs)
    else:
        assert rt._cfg_for(segs[-1].length, segs) == expected


def test_dit_cfg_selection_counts_the_longest_prompt():
    # 2048² with a 9216-token condition fits the eager cache with these prompts, not with a negative prompt 3000 tokens longer
    rt = _fake_dit(xxl=True)
    pad, shapes = _layout([30, 40], [(96, 96)], (128, 128))
    segs = build_segments(pad, shapes)
    assert rt._cfg_for(segs[-1].length, segs) == "eager"
    rt.prompt_lens = [70, 3070]
    assert rt._cfg_for(segs[-1].length, segs) == "xxl"


def test_xxl_pieces_and_block_orders():
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_dit import RBLNDiTKV
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_paged import BlockLayout

    rt = _fake_dit(xxl=True)
    pg = rt.paged
    P, part = pg.P, pg.part
    N = 16385  # odd: A gets one row more than B
    rows = torch.arange(N, dtype=torch.float32)[:, None].repeat(1, 4)
    rope = torch.arange(N)[None].repeat(3, 1)
    x, r, (nA, nB) = RBLNDiTKV._piece_rows(rows, rope, P)
    assert (nA, nB) == (8193, 8192)
    assert torch.equal(torch.cat([x[:nA], x[P : P + nB]]), rows)
    assert torch.equal(torch.cat([r[:, :nA], r[:, P : P + nB]], 1), rope)
    # a text chunk, then a split block: A goes in the text's block, B in the next one
    lay = BlockLayout(1, 10, part)
    s = lay.place(64)
    sa, sb = lay.place(P), lay.place(P)
    assert (s // part, sa // part, sb // part) == (1, 1, 2)
    key, calls, _ = pg._layer_calls("block", {"before": lay.blocks_upto(sa)[:-1], "sa": sa, "sb": sb})
    assert key == "piece"
    (w_bt, w_seq, _), (b_bt, b_seq, _), (a_bt, a_seq, _) = calls
    assert w_bt == [1] and b_bt == [1, 2] and a_bt == [2, 1]  # A written; B sees A then B; A sees B then A
    assert w_seq == sa % part and b_seq == part + sb % part and a_seq == part + sa % part


def test_dit_target_goes_after_the_prefixes():
    rt = _fake_dit()
    assert rt._tgt_start("flash", 4352, 600) == 600  # fits in the current partition
    assert rt._tgt_start("flash", 4352, 15000) == 16384  # would cross a partition boundary: next partition
    assert rt._tgt_start("eager", 16896, 15000) == 15000
    with pytest.raises(ValueError):
        rt._tgt_start("eager", 16896, 32256 - 16000)  # no room left in the eager cache


class _FakeLayerRuntime:
    """Stands in for a layer graph: outputs (2·x, x + 1); `bad` runs write a wrong value (a platform disagreement)."""

    def __init__(self, bad=()):
        self.calls, self.bad = 0, set(bad)

    def run(self, x, out=None):
        self.calls += 1
        res = [2 * x, x + 1]
        if self.calls in self.bad:
            res[0] = res[0] + 1
        if out is None:
            return res
        for o, r in zip(out, res, strict=True):
            o.copy_(r)
        return out


def _paged_for_layer_tests():
    from optimum.rbln.diffusers.pipelines.qwenimage21.dit_paged import BLOCKS_PER_LAYER, PART, PIECE
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_aux import run_verified
    from optimum.rbln.diffusers.pipelines.qwenimage21.runtime_paged import PagedDiT

    rec = {"l0": 0, "l1": 32, "n_layers": 32, "blocks_per_layer": BLOCKS_PER_LAYER, "part": PART, "piece": PIECE}

    def run_group(rt, args, verify, multi=False):
        o = run_verified(lambda: rt.run(*args), key="dit")
        return list(o) if multi else o

    return PagedDiT([(rec, "")], [0], run_group)


def test_xxl_layer_calls_reuse_output_buffers():
    pg = _paged_for_layer_tests()
    rt = _FakeLayerRuntime()
    pg._sigs[id(rt)] = (((4,), torch.float32), ((4,), torch.float32))
    x = torch.arange(4.0)
    h1, _ = pg._layer(rt, (x,))
    torch.testing.assert_close(h1, 2 * x)
    h2, _ = pg._layer(rt, (h1,))  # the input's buffer set is not written by the next call
    torch.testing.assert_close(h2, 4 * x)
    assert h2.data_ptr() != h1.data_ptr()
    h3, _ = pg._layer(rt, (h2,))
    torch.testing.assert_close(h3, 8 * x)
    assert len(pg._bufs[pg._sigs[id(rt)]]) == 3  # three rotating sets cover a chain of calls


def test_xxl_layer_call_disagreement_falls_back_and_deferred_check_recomputes():
    pg = _paged_for_layer_tests()
    rt = _FakeLayerRuntime(bad={2})  # the second run of the first call disagrees
    pg._sigs[id(rt)] = (((4,), torch.float32), ((4,), torch.float32))
    x = torch.arange(4.0)
    h, _ = pg._layer(rt, (x,))
    torch.testing.assert_close(h, 2 * x)  # recomputed through run_group (runs until two agree)
    rt2 = _FakeLayerRuntime(bad={1})  # deferred: the disagreement is found by _settle
    pg._sigs[id(rt2)] = pg._sigs[id(rt)]
    h, _ = pg._layer(rt2, (x,), defer=True)
    fixed = pg._settle(None)
    assert fixed is not None
    torch.testing.assert_close(fixed[0], 2 * x)
    assert pg._settle(None) is None  # nothing pending any more


@pytest.mark.parametrize("dit_cfg, kind, expected", [("eager", "dec", 4), ("xxl", "dec", 8), ("eager", "enc", 2)])
def test_vae_tile_count_depends_on_the_request_only(monkeypatch, dit_cfg, kind, expected):
    # chains with other tile heights round differently, so the free memory at the call must not pick the tile count
    from optimum.rbln.diffusers.pipelines.qwenimage21 import runtime_aux as A

    largest = {"dec": {4: 5.2, 6: 4.0, 8: 3.4}, "enc": {1: 4.3, 2: 2.7, 3: 1.6}}[kind]  # chiplet-0 GiB (2048² sizes)
    prog = V.DECODER_PROGRAM if kind == "dec" else V.ENCODER_PROGRAM
    vae = object.__new__(A.VAERuntime)
    vae.dit_cfg = lambda: dit_cfg
    vae._chain = lambda k, H, W, min_tiles=1: (
        f"t{min(n for n in largest if n >= min_tiles)}",
        prog,
        min(n for n in largest if n >= min_tiles),
        {},
    )
    vae._chiplet0_gib = lambda files: [largest[int(os.path.basename(os.path.dirname(files[0]))[1:])]]
    vae._load = lambda files: [None] * len(files)
    monkeypatch.setattr(A, "chiplet0_free", lambda device: pytest.fail("the tile count must not read the free memory"))
    assert vae._fitting(kind, 128, 128, (6, 8) if kind == "dec" else (2, 3, 4, 6))[2] == expected
