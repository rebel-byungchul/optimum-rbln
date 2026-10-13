# Qwen-Image-2.1 (Turbo) on RBLN CR13

`RBLNQwenImage21Pipeline` runs every model computation of
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) on RBLN NPUs: the Qwen3-VL text encoder
(vision tower and language model), the DiT, and the VAE encoder and decoder. The host only tokenizes, preprocesses
images, builds integer metadata (positions, cache slots, masks) and runs the scheduler update. It subclasses the
diffusers `QwenImage21Pipeline`, so the call arguments and the prompt templates, scheduler and post-processing are the
original ones.

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

VAE graphs are compiled at the exact image size the first time a size is used (30 s to 5 min) and cached under
`<save-dir>/vae`. Pass `rbln_config={"vae_sizes": [[h, w], ...]}` at export to precompile sizes.

## Supported features

| Feature | Status |
|---|---|
| Text-to-image, all aspect-ratio presets of the model card (2048², 2400×1792, 1792×2400, 2528×1696, 1696×2528, 2752×1536, 1536×2752) and 1024-class sizes | supported |
| Transparent (RGBA) output and RGBA inputs | supported |
| Single-image editing, local edits through annotated images | supported (1024 and 2048 class outputs) |
| Multi-reference editing | up to 10 references for 1024-class outputs; up to 3 references for 2048-class outputs |
| True CFG with a negative prompt (`true_cfg_scale > 1`) | supported |
| `use_kv_cache=True/False` | supported (same prefix-cache semantics as the original) |
| Batch size > 1 | not supported |
| `output_resolution` above the 1024 class (condition images larger than 4352 tokens) | not supported |

Measured end-to-end latency (pipeline call, 8 steps, second run of the same request; two CR13 devices):

| Case | Output | e2e (s) | DiT 8 steps (s) | VAE (s) |
|---|---|---|---|---|
| Text-to-image | 1024×1024 | 2.81 | 2.32 | 0.33 |
| Text-to-image, RGBA | 1024×1024 | 2.84 | 2.35 | 0.33 |
| Text-to-image + CFG | 1024×1024 | 5.56 | 5.00 | 0.34 |
| Edit, 1 reference | 1024×1024 | 3.80 | 2.64 | 0.43 |
| Edit, 2 references | 1024×1024 | 4.81 | 2.92 | 0.52 |
| Edit, 10 references | 1024×1024 | 15.26 | 5.96 | 1.30 |
| Text-to-image (model card example, 1.7k-token prompt) | 1696×2528 | 22.70 | 14.75 | 7.37 |
| Text-to-image | 2048×2048 | 19.93 | 14.32 | 5.26 |
| Edit (model card example) | 2048×2048 | 24.94 | 15.71 | 8.15 |

The first request after a switch between the 1024 class and larger outputs also loads the other DiT configuration
(16-35 s), see below.

## How it works

- **DiT** (`dit_kv.py`, `runtime_dit.py`): text and condition-image tokens are modulated with `t = 0` in Qwen-Image 2.1,
  so their keys and values do not change across steps. They are written once per request into device-resident paged
  KV caches (`rbln_custom_ops.paged_(flash_)causal_attn_prefill`, causal for text, bidirectional inside an image block,
  which reproduces the original block-causal mask). Every denoising step then recomputes only the target tokens,
  exactly like the original `kv_cache_mode` extract/cached. Each configuration is a set of "unified" compiled models
  (several `(kind, token bucket)` executors that share weights and caches):
  - `flash` (5 × 16384 partitions, buckets 512/4352/4352): the 1024 class, up to 10 references.
  - `eager` (one 32256-token block, buckets 512/4352/16896): larger targets up to the 2048 class.
  Only one configuration is resident; a request that needs the other one swaps them.
- **Text encoder** (`te_kv.py`, `runtime_te.py`): the Qwen3-VL language model runs chunk by chunk over a paged KV cache
  (causal, so chunking gives the same result); image features and deepstack features are injected inside the graph, and
  mRoPE tables are gathered inside the graph. The vision tower uses patch buckets with a key mask (`vision.py`).
- **VAE** (`vae.py`): the Wan-style 3D VAE is rewritten in 2D for single frames. The decoder is a chain of graphs; at
  the 2048 class its high-resolution tail runs in row tiles with halos. The tail only has local operations, so tiling
  gives the same result (verified bit-exact on CR13).

### CR13 workarounds

- Precision: graphs are compiled from fp32 modules, which CR13 computes in dlfloat16. SiLU and GELU-tanh use exp-based
  formulas, and the timestep sinusoid uses an exact table lookup (`ops.ExactTimestepProj`), because the default kernels
  lose accuracy (SiLU: 6.6 % relative error per token on the text encoder MLP).
- Memory: CR13 DRAM is split into 4 chiplets of 35 GiB, and chiplet 0 receives most of the scratch memory. Scratch
  (Init/Intermediate) is allocated per executor and per layer, so the 2048-class DiT is split into 8-layer models that
  are placed over two devices by chiplet-0 budget, and large VAE graphs are loaded per call when they do not fit next
  to the DiT. A load that fails half way does not return its memory, so these decisions are made up front.
- Compiler: high-dimensional views, channel gathers, `pixel_unshuffle` and boolean ops are rewritten; channel-reducing
  residual adds are cut out of graphs; flash chunks must lie inside one partition and eager caches must stay below 32768
  tokens.

## Known issues

- 2048-class outputs are not bit-reproducible: repeating an identical call to an 8-layer eager DiT model sometimes gives a
  different result (2-4 distinct outputs in 20 calls, relative difference up to 7.6e-3 per call, the dlfloat rounding
  level). It reproduces with a single compiled model and fixed inputs, while 1-layer graphs and the 1024-class (flash)
  models are bit-reproducible, so it points to the compiler/runtime. Two runs of the same 2048-class request differ by
  about 30 dB PSNR with the same content, which is the size of the gap between this port and CPU bf16 at 1024².

## Tests

`tests/unit/cpu/test_qwenimage21.py` checks on CPU that the rewritten pieces match diffusers: the RoPE gather and
segment layout against `QwenImage21Rope`, the exact timestep projection (bit-exact), the cache layout, the 2D VAE
decoder/encoder against `AutoencoderKLQwenImage21`, and the DiT layer-group placement. It is skipped when diffusers has
no `QwenImage21Pipeline`.
