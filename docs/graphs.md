# The graph contract

Everything a runtime needs to run FastWan is a directory of `.tflite` graphs
and one `manifest.json`. This file is the contract between the exporter
(`python/fastwan_litert/export.py`) and the four runtimes (Python, web,
Android, iOS). A runtime that follows it runs any export; an exporter that
follows it feeds any runtime.

Conventions: every tensor is `float32` unless stated; shapes are exact (no
batch dimension); row-major; `d = heads · head_dim` (1536), `H = heads`
(12), `hd = head_dim` (128), `L = text_len` (512), `C` = the row chunk of
the export, `S` = tokens of a clip, `Cq` = the attention query chunk.
Inputs are bound **by name** (the LiteRT signature input names below, which
`litert-torch` takes from the module's keyword arguments). Outputs are
bound **by position**: the converter names them `output_0`, `output_1`, …
in the order the tables below list them.

## `manifest.json`

```json
{
  "format": "fastwan-litert/1",
  "model": "fastwan-1.3b",                 // or "tiny"
  "config": {
    "patch": 2, "heads": 12, "head_dim": 128, "in_channels": 16, "out_channels": 16,
    "text_dim": 4096, "text_len": 512, "freq_dim": 256, "ffn_dim": 8960, "layers": 30,
    "eps": 1e-6, "rope_theta": 10000.0
  },
  "tae": { "latent_channels": 16, "widths": [256, 128, 64, 64], "tgrow": [1, 2, 2],
           "mem_blocks": 3, "patch": 1, "denormalize": false },
  "schedule": { "timesteps": [1000, 757, 522], "sigmas": [1.0, 0.757, 0.522, 0.0] },
  "weights": "int4",                       // f32 | f16 | int8 | int4 (block linears)
  "chunk": 1024,                           // C
  "graphs": {
    "time_mods":   "time_mods.tflite",
    "text_embed":  "text_embed.tflite",
    "patch_embed": "patch_embed.tflite",
    "head":        "head.tflite",
    "blocks": [ { "qkv": "block_00_qkv.tflite", "post": "block_00_post.tflite",
                  "kv": "block_00_kv.tflite", "bytes": 94371840 }, ... ]
  },
  "attention": [ { "tokens": 32760, "q_chunk": 256, "heads": 12,
                   "file": "attn_s32760_q256_h12.tflite" }, ... ],
  "tae_chunks": [ { "frames": 3, "latent_h": 60, "latent_w": 104,
                    "file": "tae_f3_60x104.tflite" }, ... ],
  "sizes": [ { "width": 832, "height": 480, "frames": 81 }, ... ],
  "files": { "block_00_qkv.tflite": { "bytes": 62914560, "sha256": "..." }, ... }
}
```

`sizes` lists the clip geometries this export has attention and TAEHV
graphs for. A runtime picks the `attention` entry whose `tokens` equals the
clip's token count and the `tae_chunks` entry whose `latent_h/w` equal the
clip's latent size; a missing entry is an error ("this export has no graphs
for 640×352"), never a silent fallback. `files` is for integrity and for
a page's download progress.

## Graphs

### `time_mods`

Timestep conditioning for one DMD step: Diffusers `Timesteps(freq_dim,
flip_sin_to_cos=True, downscale_freq_shift=0)` → `time_embedder` (linear,
SiLU, linear) → `time_proj` (SiLU, linear) → `+ scale_shift_table` per
block; and the output head's `[2, d]` table `scale_shift_table + temb`.

| | name | shape | note |
|---|---|---|---|
| in | `t` | `[1]` | the DiT timestep, e.g. `757.0` |
| out | `block_mods` | `[layers, 6, d]` | slot order shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn |
| out | `head_mods` | `[2, d]` | row 0 shift, row 1 scale |

### `text_embed`

`text_embedder`: linear, GELU (tanh approximation), linear.

| | name | shape |
|---|---|---|
| in | `embeds` | `[L, text_dim]` UMT5 output, zero past the prompt |
| out | `ctx` | `[L, d]` |

### `block_N_kv`

Block `N`'s cross-attention keys and values of the prompt: `to_k`, `to_v`
of `attn2`, `norm_k` (RMSNorm across all heads, `eps`) on the keys, split
into heads.

| | name | shape |
|---|---|---|
| in | `ctx` | `[L, d]` |
| out | `k2` | `[H, L, hd]` |
| out | `v2` | `[H, L, hd]` |

### `patch_embed`

`patch_embedding` (Conv3d `(1, p, p)` as a linear over the patch features).

| | name | shape |
|---|---|---|
| in | `patches` | `[C, p²·in_channels]`, feature index `c·p² + kh·p + kw` |
| out | `h` | `[C, d]` |

### `block_N_qkv`

`n = LN(h)·(1 + scale_msa) + shift_msa` (no affine), `qkv = n·W_qkvᵀ + b`,
`q = rope(rms(q)·norm_q)`, `k = rope(rms(k)·norm_k)`, `v`. RMSNorm is
across all `d` channels (Diffusers `rms_norm_across_heads`). RoPE rotates
interleaved pairs `(x[2i], x[2i+1])` of every head by angle `i` of the row.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `mods` | `[6, d]` (`block_mods[N]`) |
| in | `cos` | `[C, hd/2]` |
| in | `sin` | `[C, hd/2]` |
| out | `q` | `[C, H, hd]` |
| out | `k` | `[C, H, hd]` |
| out | `v` | `[C, H, hd]` |

### `attn_sS_qCq_hH`

Dense softmax attention, scale `1/√hd`, no mask, heads merged on the way
out. Weight-free; shared by every block and step.

| | name | shape |
|---|---|---|
| in | `q` | `[Cq, H, hd]` |
| in | `k` | `[S, H, hd]` |
| in | `v` | `[S, H, hd]` |
| out | `o` | `[Cq, H·hd]` |

### `block_N_post`

`h += o·W_outᵀ·gate_msa`; `n = norm2(h)` (LayerNorm with affine);
cross-attention `attn(rms(n·W_q2ᵀ)·norm_q2, k2, v2)` over `L` keys, `W_out2`,
`h += ·`; `n = LN(h)·(1 + scale_ffn) + shift_ffn`; `h += ffn(n)·gate_ffn`
with `ffn = linear(gelu_tanh(linear(n)))`.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `o` | `[C, d]` (the attention output of these rows) |
| in | `mods` | `[6, d]` |
| in | `k2` | `[H, L, hd]` |
| in | `v2` | `[H, L, hd]` |
| out | `h_out` | `[C, d]` |

### `head`

`LN(h)·(1 + scale) + shift` with `head_mods`, then `proj_out`.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `mods` | `[2, d]` (`head_mods`) |
| out | `patches` | `[C, p²·out_channels]`, feature index `(kh·p + kw)·c + c'` (Diffusers unpatchify order) |

### `tae_fF_HxW`

One TAEHV chunk of `F` latent frames at latent size `H × W`. Inside:
LightTAE denormalisation if `tae.denormalize` (`z·std + mean`), `tanh(z/3)·3`,
`conv 16→n0, relu`, three stages of `3 × MemBlock → ×2 nearest upsample →
TGrow → conv`, `relu, conv n3→3p², pixel-shuffle p, clamp [0, 1]`.
`MemBlock(x, past) = relu(conv(relu(conv(relu(conv(cat[x, past]))))) + x)`
where `past` is the input shifted by one frame: `cat([mem, x[:-1]])` along
frames, `mem` the block's last input frame of the previous chunk (zeros for
the first chunk). Stage `s` MemBlocks run on `F · Π tgrow[<s]` frames of
`widths[s]` channels at `(H·2^s) × (W·2^s)`.

| | name | shape |
|---|---|---|
| in | `z` | `[F, 16, H, W]` frame-major latent |
| in | `mem_0` … `mem_8` | `[1, widths[s], H·2^s, W·2^s]`, `s = i / mem_blocks` (memory `i` belongs to MemBlock `i mod mem_blocks` of stage `s`) |
| out | `rgb` | `[4F, 3, 8pH, 8pW]` in `[0, 1]` |
| out | `mem_out_0` … `mem_out_8` | as the inputs: pass to the next chunk |

The clip's first `3` frames (`t_upscale − 1`) are priming frames and are
dropped by the host; a clip of `T` latent frames decodes to `4T − 3`.

## The host loop

```
σ = schedule.sigmas; x = rng(seed).normal(16·T·Hl·Wl)            # latent [16, T, Hl, Wl]
ctx = text_embed(embeds); k2[N], v2[N] = block_N_kv(ctx)          # once per prompt
cos, sin = rope_table(grid, start_frame=0)                        # [S, 64]
for i, t in enumerate(timesteps):
    block_mods, head_mods = time_mods([t])
    h = concat(patch_embed(chunk) for chunk in pad(patchify(x), C))   # [S, d] (padded rows dropped)
    for N in blocks:
        q, k, v = concat(block_N_qkv(h[c], block_mods[N], cos[c], sin[c]) for c in chunks)   # [S, H, hd]
        o = concat(attn(q[cq], k, v) for cq in q_chunks(S, Cq))                               # [S, d]
        h = concat(block_N_post(h[c], o[c], block_mods[N], k2[N], v2[N]) for c in chunks)
    v_pred = unpatchify(concat(head(h[c], head_mods) for c in chunks))
    x = x − σ[i]·v_pred;  if i + 1 < steps: x = (1 − σ[i+1])·x + σ[i+1]·rng.normal(...)
mem = zeros; frames = []
for chunk in latent frames of x in groups of F (last group zero-padded to F):
    rgb, mem = tae(chunk, mem); frames += rgb (dropping padded frames' output, then the 3 priming frames)
```

Rules every runtime follows:

- **Padding.** The last row chunk is zero-padded to `C` and its padded rows'
  outputs are discarded; the queries are zero-padded to a multiple of `Cq`
  the same way; keys and values are never padded (`S` exact). The last TAEHV
  chunk is zero-padded to `F` latent frames and the decoded frames past
  `4 · real` are dropped. Padding never changes a real output: every per-row
  graph is row-independent and TAEHV's memory only looks backwards.
- **Noise order.** Draw the initial latent first, then, in step order, the
  re-noise of every step but the last, all from one generator seeded with
  the request's seed. Fresh noise is drawn in `[16, T, Hl, Wl]` order.
- **No implicit fallbacks.** A geometry without graphs, a manifest of
  another `format`, or a graph whose signature does not match this file is
  an error with the mismatch in its message.
- **Events.** Runtimes report `prepared {ms, tokens}`, one `step {index, total, ms}` per
  DMD step and `frames {first, count, width, height, rgb}` per TAEHV chunk
  (times in milliseconds; `rel` in the fixture check is `max|Δ| / max|expected|`),
  so a page can play frames while the clip is still decoding.

## Weight formats

The exporter writes f32 graphs and then quantizes the **block linears**
(`attn1.*`, `attn2.*`, `ffn.*`, `to_gate_compress`: 97 % of the DiT) with
`ai-edge-quantizer` as `--weights` says: `f16` (half the download), `int8`
(weight-only, per-channel), `int4` (weight-only, blockwise 32). Activations
stay f32 in the graph; a GPU delegate may run them in f16. Everything
outside the blocks (patch / text / time embedders, `proj_out`, TAEHV)
stays f32 or f16. The `weights` field of the manifest records the choice.

## The tiny fixture

`fastwan-litert export --model tiny --out fixtures/tiny` writes the tiny
model (`WanConfig.tiny()`: 2 layers, 2 heads of 12, `TaeConfig.tiny()`) with
deterministic synthetic weights, its graphs at `C = 16`, `Cq = 8`, one
geometry (`32 × 48 × 9` → latent `(3, 4, 6)`, 18 tokens), and
`expected.safetensors`: the denoised latent and decoded frames for seed 7
and a fixed synthetic prompt, computed by the PyTorch modules. Every
runtime's self-test (`check`) runs the fixture through its own loop and
compares against those tensors (rel ≤ 1e-3 on CPU; a GPU delegate in f16
is judged at 2e-2).
