"""Export: Diffusers checkpoint (or synthetic weights) → the graph directory of
`docs/graphs.md`.

Every graph module is converted with `litert-torch` at fixed shapes; the block
graphs are then quantized with `ai-edge-quantizer` as `--weights` says. The
manifest records shapes, files, sizes and hashes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import TaeConfig, WanConfig
from .geometry import Geometry
from .modules import Attention, TaeHv, WanDit
from .runtime import FORMAT
from .schedule import DmdSchedule
from .weights import Synthetic, WeightSource, load_dit, load_tae

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

WEIGHT_FORMATS = ("f32", "f16", "int8", "int4")


class TaeChunkGraph(torch.nn.Module):
    """`TaeHv` with the nine memories as named inputs (`mem_0` … `mem_8`), so
    the converter names them as the contract says."""

    def __init__(self, tae: TaeHv):
        super().__init__()
        assert 3 * tae.cfg.mem_blocks == 9, "the contract names nine memories"
        self.tae = tae

    def forward(self, z, mem_0, mem_1, mem_2, mem_3, mem_4, mem_5, mem_6, mem_7, mem_8):
        return self.tae(z, mem_0, mem_1, mem_2, mem_3, mem_4, mem_5, mem_6, mem_7, mem_8)


def _convert(module: torch.nn.Module, kwargs: dict[str, torch.Tensor], out: Path, f16: bool) -> None:
    import litert_torch

    flags = None
    if f16:
        import tensorflow as tf

        # nested keys: litert-torch walks the dict (`converter.target_spec.supported_types`); a dotted
        # key sets a stray attribute and `Optimize.DEFAULT` alone quantizes to dynamic-range int8
        flags = {"optimizations": [tf.lite.Optimize.DEFAULT], "target_spec": {"supported_types": [tf.float16]}}
    # `strict_export=True`: one torch.export path, so an export is reproducible byte for byte.
    edge = litert_torch.convert(module.eval(), sample_kwargs=kwargs, strict_export=True, _ai_edge_converter_flags=flags)
    edge.export(str(out))


def _block_job(args) -> dict:
    """Worker: export one block's `kv`, `qkv` and `post` graphs from a fresh source."""
    cfg_json, src_spec, l, out_dir, chunk, weights = args
    import warnings

    warnings.filterwarnings("ignore")
    from .modules import BlockKV, BlockPost, BlockQkv
    from .weights import load_block, open_source

    cfg = WanConfig.from_json(cfg_json)
    src = open_source(src_spec)
    out_dir = Path(out_dir)
    d, H, hd, L = cfg.dim, cfg.heads, cfg.head_dim, cfg.text_len
    qkv, kv, post = BlockQkv(cfg).eval(), BlockKV(cfg).eval(), BlockPost(cfg).eval()
    load_block(l, qkv, kv, post, src, cfg)
    entry = {}
    for kind, module, kw in (
        ("kv", kv, {"ctx": torch.randn(L, d)}),
        ("qkv", qkv, {"h": torch.randn(chunk, d), "mods": torch.randn(6, d), "cos": torch.rand(chunk, hd // 2), "sin": torch.rand(chunk, hd // 2)}),
        ("post", post, {"h": torch.randn(chunk, d), "o": torch.randn(chunk, d), "mods": torch.randn(6, d), "k2": torch.randn(H, L, hd), "v2": torch.randn(H, L, hd)}),
    ):
        name = f"block_{l:02d}_{kind}.tflite"
        t0 = time.time()
        _convert(module, kw, out_dir / name, weights == "f16")
        if weights in ("int8", "int4"):
            _quantize(out_dir / name, weights)
        entry[kind] = name
        entry[f"{kind}_secs"] = round(time.time() - t0, 1)
    return {"l": l, **entry}


def _quantize(path: Path, weights: str) -> None:
    """Weight-only int8 / int4 on the block linears of one graph file, in place."""
    from ai_edge_quantizer import quantizer, recipe

    rec = recipe.weight_only_wi8_afp32() if weights == "int8" else recipe.weight_only_wi4_afp32()
    q = quantizer.Quantizer(str(path), rec)
    res = q.quantize()
    tmp = path.with_suffix(".q.tflite")
    res.export_model(str(tmp))
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export(
    out_dir: Path,
    cfg: WanConfig,
    tcfg: TaeConfig,
    dit_src: WeightSource,
    tae_src: WeightSource,
    *,
    src_spec,
    model_name: str,
    weights: str = "f16",
    chunk: int = 1024,
    q_chunk: int = 256,
    sizes: list[Geometry],
    tae_frames: int = 3,
    schedule: DmdSchedule | None = None,
    jobs: int = 1,
    log=print,
) -> dict:
    if weights not in WEIGHT_FORMATS:
        raise ValueError(f"--weights must be one of {WEIGHT_FORMATS}")
    for g in sizes:
        g.validate(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sched = schedule or DmdSchedule.fastwan_1_3b()
    torch.manual_seed(0)
    dit = WanDit(cfg, with_blocks=False).eval()
    load_dit(dit, dit_src)
    tae = TaeHv(tcfg).eval()
    load_tae(tae, tae_src)
    d, H, hd, L = cfg.dim, cfg.heads, cfg.head_dim, cfg.text_len
    f16 = weights != "f32"  # non-block graphs: f16 unless the export is all-f32
    files: dict[str, dict] = {}

    def done(name: str, t0: float):
        p = out_dir / name
        files[name] = {"bytes": p.stat().st_size, "sha256": _sha(p)}
        log(f"  {name:32s} {p.stat().st_size / 1e6:8.1f} MB  {time.time() - t0:5.1f} s")

    graphs: dict = {}
    t0 = time.time()
    _convert(dit.time, {"t": torch.tensor([757.0])}, out_dir / "time_mods.tflite", False)
    graphs["time_mods"] = "time_mods.tflite"
    done("time_mods.tflite", t0)
    t0 = time.time()
    _convert(dit.text, {"embeds": torch.randn(L, cfg.text_dim)}, out_dir / "text_embed.tflite", f16)
    graphs["text_embed"] = "text_embed.tflite"
    done("text_embed.tflite", t0)
    t0 = time.time()
    _convert(dit.patch, {"patches": torch.randn(chunk, cfg.patch_features_in())}, out_dir / "patch_embed.tflite", f16)
    graphs["patch_embed"] = "patch_embed.tflite"
    done("patch_embed.tflite", t0)
    t0 = time.time()
    _convert(dit.head, {"h": torch.randn(chunk, d), "mods": torch.randn(2, d)}, out_dir / "head.tflite", f16)
    graphs["head"] = "head.tflite"
    done("head.tflite", t0)

    job_args = [(cfg.to_json(), src_spec, l, str(out_dir), chunk, weights) for l in range(cfg.layers)]
    results = []
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(jobs) as pool:
            for r in pool.imap_unordered(_block_job, job_args):
                results.append(r)
                log(f"  block {r['l']:02d}: kv {r['kv_secs']} s, qkv {r['qkv_secs']} s, post {r['post_secs']} s")
    else:
        for a in job_args:
            r = _block_job(a)
            results.append(r)
            log(f"  block {r['l']:02d}: kv {r['kv_secs']} s, qkv {r['qkv_secs']} s, post {r['post_secs']} s")
    results.sort(key=lambda r: r["l"])
    blocks = []
    for r in results:
        for k in ("kv", "qkv", "post"):
            pth = out_dir / r[k]
            files[r[k]] = {"bytes": pth.stat().st_size, "sha256": _sha(pth)}
        blocks.append({"kv": r["kv"], "qkv": r["qkv"], "post": r["post"], "bytes": sum(files[r[k]]["bytes"] for k in ("kv", "qkv", "post"))})
    graphs["blocks"] = blocks

    attention = []
    tae_chunks = []
    seen_s: set[int] = set()
    seen_hw: set[tuple[int, int]] = set()
    for g in sizes:
        s = g.tokens(cfg)
        if s not in seen_s:
            seen_s.add(s)
            cq = min(q_chunk, s)
            name = f"attn_s{s}_q{cq}_h{H}.tflite"
            t0 = time.time()
            _convert(Attention(), {"q": torch.randn(cq, H, hd), "k": torch.randn(s, H, hd), "v": torch.randn(s, H, hd)}, out_dir / name, False)
            done(name, t0)
            attention.append({"tokens": s, "q_chunk": cq, "heads": H, "file": name})
        _, lh, lw = g.latent()
        if (lh, lw) not in seen_hw:
            seen_hw.add((lh, lw))
            f = min(tae_frames, g.latent()[0])
            name = f"tae_f{f}_{lh}x{lw}.tflite"
            t0 = time.time()
            kw = {"z": torch.randn(f, tcfg.latent_channels, lh, lw)}
            for i, shape in enumerate(tae.memory_shapes(lh, lw)):
                kw[f"mem_{i}"] = torch.zeros(shape)
            _convert(TaeChunkGraph(tae), kw, out_dir / name, f16)
            done(name, t0)
            tae_chunks.append({"frames": f, "latent_h": lh, "latent_w": lw, "file": name})

    manifest = {
        "format": FORMAT,
        "model": model_name,
        "config": cfg.to_json(),
        "tae": tcfg.to_json(),
        "schedule": sched.to_json(),
        "weights": weights,
        "chunk": chunk,
        "graphs": graphs,
        "attention": attention,
        "tae_chunks": tae_chunks,
        "sizes": [g.to_json() for g in sizes],
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f["bytes"] for f in files.values())
    log(f"wrote {out_dir / 'manifest.json'}: {len(files)} graphs, {total / 1e6:.1f} MB")
    return manifest


# ---- the tiny fixture -------------------------------------------------------------

TINY_GEOMETRY = Geometry(32, 48, 9)  # latent (3, 4, 6): 18 tokens
TINY_SEED = 7


def tiny_prompt(cfg: WanConfig) -> np.ndarray:
    """A fixed synthetic prompt: the first 5 rows set, the rest zero (as a
    real zero-padded UMT5 output)."""
    from .rng import Rng

    e = np.zeros((cfg.text_len, cfg.text_dim), np.float32)
    e[:5] = Rng(1234).normal(5 * cfg.text_dim).reshape(5, cfg.text_dim)
    return e


def export_tiny(out_dir: Path, weights: str = "f32", log=print) -> dict:
    """The tiny synthetic model plus `expected.safetensors` (`docs/graphs.md`)."""
    from safetensors.numpy import save_file

    from . import reference

    cfg, tcfg = WanConfig.tiny(), TaeConfig.tiny()
    src = Synthetic(0)
    manifest = export(
        out_dir, cfg, tcfg, src, src, src_spec=("synthetic", 0), model_name="tiny", weights=weights, chunk=16, q_chunk=8,
        sizes=[TINY_GEOMETRY], tae_frames=2, log=log,
    )
    dit = WanDit(cfg).eval()
    load_dit(dit, src)
    tae = TaeHv(tcfg).eval()
    load_tae(tae, src)
    embeds = tiny_prompt(cfg)
    latent = reference.denoise(dit, DmdSchedule.fastwan_1_3b(), embeds, TINY_GEOMETRY, TINY_SEED)
    frames = reference.decode(tae, latent, 2)
    save_file(
        {"prompt_embeds": embeds, "latent": latent.astype(np.float32), "frames": frames},
        str(Path(out_dir) / "expected.safetensors"),
        metadata={"seed": str(TINY_SEED), "geometry": str(TINY_GEOMETRY)},
    )
    log(f"wrote expected.safetensors: latent {latent.shape}, frames {frames.shape}")
    return manifest
