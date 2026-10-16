# Qwen-Image-2.1 (Turbo) on RBLN CR13

`RBLNQwenImage21Pipeline` runs every model computation of
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) on RBLN NPUs: the Qwen3-VL text encoder
(vision tower and language model), the DiT, and the VAE encoder and decoder. The host only tokenizes, preprocesses
images, builds integer metadata (positions, cache slots, masks) and runs the scheduler update (in fp32). It subclasses the
diffusers `QwenImage21Pipeline`, so the call arguments, prompt templates, scheduler and post-processing are the original
ones.

## Requirements

- Two CR13 devices (default `dit_device=[0, 1]`, `aux_device=0`).
- `diffusers>=0.41` (currently the main branch, which has `QwenImage21Pipeline`) and `transformers>=5.17`. These are newer
  than the versions pinned by this repository, so install them on top:
  `pip install git+https://github.com/huggingface/diffusers.git "transformers>=5.17"`.
- Tested with rebel-compiler 0.11.3.dev606, torch 2.11, diffusers main `916a5791`, transformers 5.17.0.

## Usage

```bash
# compile once (DiT, text encoder and vision graph banks), then reuse the directory
python run_qwenimage21.py --model-id Qwen/Qwen-Image-2.1-Turbo --save-dir qwen-image-2.1-turbo-rbln --export
python run_qwenimage21.py --model-id qwen-image-2.1-turbo-rbln --width 1696 --height 2528 --prompt "..."
python run_qwenimage21.py --model-id qwen-image-2.1-turbo-rbln --image sketch.png --width 2048 --height 2048 --prompt "..."
```

VAE graphs are compiled at the exact image size the first time a size is used (20 s to 5 min) and cached under
`<save-dir>/vae`. Pass `rbln_config={"vae_sizes": [[h, w], ...]}` at export to precompile sizes.

## Supported features

| Feature | Status |
|---|---|
| Text-to-image: 1024 class, any size up to the 2048 class, all model-card presets | supported |
| Transparent (RGBA) output and RGBA inputs | supported |
| Single-image and local (annotated) editing at 1024 and 2048 class | supported |
| Multi-reference editing | up to 36 references of the 1024 class, at 1024-class and 2048-class outputs (2048² and every preset) |
| `output_resolution` up to 2048 (2048-class condition images) | up to 9 such condition images, also next to a 2048² output |
| True CFG with a negative prompt, `use_kv_cache=True/False` | supported |
| Batches (several prompts, `num_images_per_prompt`) | supported (same result as the original: sample 0 equals a single run with the same seed) |
| Odd latent patch grids (e.g. 2528x1648) | the original transformer fails with a shape error; so does this pipeline |

Two compiler constraints measured on rebel-compiler 0.11.3.dev606 shape the DiT: paged flash attention is wrong across
cache partitions for chunks of 16384 tokens or more, and the eager paged attention cache must stay below 32768 tokens.
Requests beyond what the `flash`/`eager`/`xl` caches hold (2048-class outputs with long prefixes) run on the `xxl` DiT
(below). Its block cache holds 18 prefix blocks of 8960 tokens per layer and context (the limits above, ~160k tokens of
prefix); a longer prefix raises a `ValueError` that names the limit. With true CFG, a second context that does not fit next to the first is
rewritten on every call (same result, slower).

Measured end-to-end latency (pipeline call, 8 steps, second run of the same request, two CR13 devices; the
2048-class DiT and the text/vision/VAE graphs include the repeated runs described under "Reliability"):

| Case | Output | e2e (s) | DiT 8 steps (s) | VAE (s) |
|---|---|---|---|---|
| Text-to-image | 1024² | 2.84 | 2.37 | 0.32 |
| Text-to-image, RGBA | 1024² | 2.83 | 2.37 | 0.33 |
| Text-to-image + CFG | 1024² | 5.16 | 4.62 | 0.32 |
| Text-to-image, 2 prompts × 2 images | 4 × 1024² | 11.51 | 9.80 | 1.28 |
| Edit, 1 / 2 / 10 references | 1024² | 4.03 / 5.37 / 18.16 | 2.60 / 2.90 / 6.00 | 0.56 / 0.75 / 2.54 |
| Text-to-image, model-card presets (7) | 2048 class | 37.9 – 45.2 | 28.6 – 29.9 | 9.0 – 15.8 |
| Edit, model-card example (yacht sketch) | 2048² | 43.28 | 31.16 | 10.70 |
| Edit, 4 references (xl) | 2528x1632 | 76.03 | 40.58 | 19.87 |
| Edit, `output_resolution=1536` | 1376x1728 | 118.28 | 37.84 | 11.46 |
| Edit, 4 / 6 references (xxl) | 2048² | 123.19 / 123.81 | 67.97 / 66.60 | 17.05 / 18.09 |
| Edit, `output_resolution=2048` (yacht sketch, xxl) | 3456x1216 | 149.75 | 64.68 | 21.35 |
| Edit, 2 / 3 / 6 references at `output_resolution=2048` (xxl) | 2048² | 151.75 / 172.14 / 246.59 | 69.97 / 74.36 / 87.78 | 16.51 / 17.98 / 25.48 |

The first request after a switch between cache configurations also loads the other DiT models (16-35 s). Requests that
use prefix-only models load them while writing the prefix (`output_resolution=1536`: 32 one-layer models holding only
the image bucket in use, ~20-30 s; 4 references at 2528x1632: 4 models, ~10 s); they are copied to local disk on first
use when there is room. The xxl DiT leaves little room for the text encoder and the large vision bucket on device 0:
these unload only as many xxl layer graphs as they need, which the next DiT call loads again (~5-10 s), and the text
encoder is loaded again for every xxl request (~15 s).

## Accuracy

Every example was also run through the original diffusers pipeline on CPU in fp32 (same prompt, inputs, seed and
initial noise) and the outputs compared (PSNR of the final image; RGBA outputs composited on white). The NPU computes in
dlfloat16/bf16, so the results are not bit-identical; for reference, the original pipeline in bf16 (the dtype of the
model card) differs from its own fp32 run by about as much or more.

| Case | Output | PSNR NPU vs original fp32 (dB) | original bf16 vs fp32 (dB) |
|---|---|---|---|
| text-to-image | 1024x1024 | 29.0 | 26.3 |
| text-to-image | 2048x2048 | 26.4 | 19.8 |
| text-to-image | 2528x1696 | 41.0 |  |
| text-to-image | 1696x2528 | 19.0 |  |
| text-to-image | 2752x1536 | 20.9 |  |
| text-to-image | 1536x2752 | 23.9 |  |
| text-to-image | 2400x1792 | 20.8 |  |
| text-to-image | 1792x2400 | 34.2 |  |
| text-to-image, RGBA | 1024x1024 | 31.4 | 22.6 |
| edit | 1024x1024 | 36.6 | 40.7 |
| edit, 2 references | 1024x1024 | 44.1 |  |
| edit | 1024x1024 | 49.7 |  |
| edit, RGBA | 1024x1024 | 50.3 |  |
| edit, 10 references | 1024x1024 | 14.6 |  |
| edit | 2048x2048 | 22.1 |  |
| edit, output_resolution 1536 | 1376x1728 | 31.4 |  |
| edit, 4 references | 2528x1632 | 31.9 |  |
| text-to-image, CFG | 1024x1024 | 36.1 |  |
| text-to-image, no KV cache | 1024x1024 | 43.6 |  |
| text-to-image | 1024x576 | 35.4 |  |
| text-to-image, batch of 4 | 1024x1024 | 35.9 / 38.3 / 46.5 / 30.8 |  |
| edit, output_resolution 2048 | 3456x1216 | 40.2 |  |
| edit, 4 references | 2048x2048 | 29.3 |  |
| edit, 6 references | 2048x2048 | 24.1 |  |
| edit, 2 references, output_resolution 2048 | 2048x2048 | 43.7 |  |
| edit, 3 references, output_resolution 2048 | 2048x2048 | 18.6 |  |
| edit, 6 references, output_resolution 2048 | 2048x2048 | 19.9 |  |

Low values come from cases whose first steps are very sensitive (a collage of 10 references, dense text): small early
differences move the layout or the strokes. The original in bf16 moves at least as much there (per-step DiT error at
the first steps, 2400x1792: NPU 6.4 / 1.2 / 0.6 % vs original bf16 8.4 / 5.9 / 4.5 %; 10 references: 6.0 / 2.6 / 3.5 %
vs 13.8 / 9.1 / 6.1 %).

Per denoising step, the DiT error of the NPU (each step started from the fp32 original's latents) is below that of the
original in bf16 in 188 of the 189 steps of the examples that have a bf16 baseline (all 27). Four precision losses were
found this way and fixed (see "CR13 workarounds"): the timestep was rounded to bf16 before the sinusoid (978.5 → 976.6
at the second step changes the DiT output by 4.5 %); the latents were kept in bf16 between steps (0.25 % rounding per
step); the xxl graphs passed their hidden state between graphs in bf16 (three times the per-step error of the other
configurations in the last steps); and the VAE encoder ran in bf16 (condition latents 1.3-1.6 % off, now
0.7-1.3 %, the original in bf16: 0.5-1.5 %). Mean PSNR over the 27 examples went from 28.8 to 31.2 dB.

## How it works

- **DiT** (`dit_kv.py`, `runtime_dit.py`): text and condition-image tokens are modulated with `t = 0` in Qwen-Image 2.1,
  so their keys and values do not change across steps. They are written once per request into device-resident paged
  KV caches (causal for text, bidirectional inside an image block, which reproduces the original block-causal mask);
  every denoising step then recomputes only the target tokens, exactly like the original `kv_cache_mode`
  extract/cached. Four cache configurations exist and one is resident at a time:
  - `flash` (5 × 16384 partitions, buckets txt 512 / img 4352 / target 4352): the 1024 class.
  - `eager` (one 32256-token block, target 16896): the 2048 class. Condition images above 4352 tokens go through
    prefix-only models loaded for the prefix phase.
  - `xl` (3 × 16384 partitions, target 16320): long prefixes; all prefix executors live in prefix-only models.
  - `xxl` (`dit_paged.py`, `runtime_paged.py`): 2048-class targets next to long prefixes. Per device, one block cache
    for all of its layers (20 blocks of 8960 tokens per layer) and a few attention graphs without weights, shared by
    every layer: the block table selects the layer's blocks, so the attention's Init scratch (it grows with query chunk
    × visible cache and is allocated per graph) exists once per device. The layer computations are per-layer weight
    graphs; q/k/v and attention outputs pass through the host. A block above one piece (the target, a 2048-class
    condition image) is split in two pieces A, B of one block each: A's keys/values are written, then B attends over
    the block order (prefix, A, B) and A over (prefix, B, A) — the original bidirectional attention with chunks below
    the flash limit. Small prefix chunks are packed so that they share the layer graph calls.
  Batches and CFG give every sample/context its own prefix region and share one target region placed right after the
  prefixes (target keys/values are rewritten on every call); the configuration is chosen for the longest prompt of the
  request (e.g. a longer negative prompt). The timestep → modulation path (sinusoid, timestep MLP, modulation and
  output-norm scale) is one small graph run once per timestep; the layer graphs take its output as an input.
- **Text encoder** (`te_kv.py`, `runtime_te.py`): the Qwen3-VL language model runs chunk by chunk over a paged KV cache
  (causal, so chunking gives the same result); image and deepstack features are injected inside the graph, and mRoPE
  tables are gathered inside the graph. Prompts are encoded one by one and padded like the original.
- **VAE** (`vae.py`, `runtime_aux.py`): the Wan-style 3D VAE is rewritten in 2D for single frames as a chain of graphs.
  Above the 1024 class everything after the decoder's mid block (and the encoder's first stages when a size does not
  compile in one piece) runs in row tiles with halos; these stages only have local operations, so the result is the
  same.

### CR13 workarounds

- **Precision**: graphs are compiled from fp32 modules, which CR13 computes in dlfloat16. SiLU and GELU-tanh use
  exp-based formulas, because the default kernels lose accuracy (SiLU: 6.6 % relative error per token on the text
  encoder MLP). Phases of up to 1000 rad cannot be computed arithmetically at this precision, so the timestep sinusoid
  is a table lookup: the fp32 timestep is passed as three bf16 pieces (`ops.split_timestep`), each located exactly with
  comparisons and looked up, and the angles are added (`ops.ExactTimestepProj`). The DiT returns fp32 noise
  predictions, so the scheduler keeps the latents in fp32 between steps, as the fp32 original does. The remaining DiT
  error comes mostly from the attention kernels (0.4-0.9 % per layer on real activations, mainly from rounding the
  attention logits). The xxl graphs pass their hidden state (residual stream) in fp32 and q/k/v and attention outputs
  in bf16 (`hidden_dtype` / `attn_dtype` in the bank record): a bf16 hidden state tripled the per-step error, fp32
  q/k/v/attention outputs added transfer time and no accuracy. The VAE encoder is compiled in fp32 (the decoder stays
  in bf16: 50.9 dB against the original).
- **Reliability**: large graphs (2048-class DiT layer groups and VAE graphs) now and then return a different result for
  the same input, sometimes with values far out of range. Every graph call except the 1024-class DiT and VAE decoder
  and the single-op xxl attention graphs (0 disagreements in 4800 layer runs / 1200 graph calls / 1728 attention calls)
  is therefore run until two results agree bit for bit (`runtime_aux.run_verified`); a 2048-class request typically
  has 1 to 4 disagreements, which are discarded. This doubles the time of the checked graphs.
- **Memory**: CR13 DRAM is split into 4 chiplets of 35 GiB and chiplet 0 receives most of the scratch memory, which is
  allocated per executor and per layer. Loads are planned against chiplet 0 (measured footprints are kept next to the
  compiled files) and components free each other's memory when needed, since a load that fails half way does not give
  its memory back. Prefix-only models share caches and weights with the resident models (compiled in the same context;
  the shared weights live in the first model's file, which is always loaded first); all executors and prefix-only
  models must use shared weights in the same way, which is why the timestep path is a separate graph. Next to a large
  resident DiT, VAE encoder and decoder graphs use more row tiles (same result) before unloading the DiT.
- **Compiler**: high-dimensional views, channel gathers, `pixel_unshuffle` and boolean ops are rewritten;
  channel-reducing residual adds are cut out of graphs; sizes that fail to compile in one piece are tiled.

## Tests

`tests/unit/cpu/test_qwenimage21.py` checks on CPU that the rewritten pieces match diffusers: the RoPE gather and
segment layout against `QwenImage21Rope`, the timestep pieces and table lookup against the exact sinusoid, the cache
layout and target placement, the cache configuration chosen for each request shape (also with a longer negative
prompt), the xxl pieces and block orders, the 2D VAE decoder/encoder and their row-tiled versions against
`AutoencoderKLQwenImage21`, and the DiT layer-group placement. It is skipped when diffusers has no
`QwenImage21Pipeline`.
