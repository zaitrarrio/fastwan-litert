"""`fastwan-litert`: export, generate, check, plan, encode, serve."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import TaeConfig, WanConfig
from .geometry import FASTWAN_480P, MOBILE_480x272, Geometry
from .schedule import DmdSchedule


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def cmd_export(a: argparse.Namespace) -> int:
    from .export import export, export_tiny

    if a.model == "tiny":
        export_tiny(Path(a.out), weights=a.weights, log=_log)
        return 0
    from .weights import SafetensorsDir, read_wan_config

    model_dir = Path(a.model_dir)
    tdir = model_dir / "transformer" if (model_dir / "transformer").is_dir() else model_dir
    cfg = read_wan_config(tdir) if (tdir / "config.json").exists() else WanConfig.fastwan_1_3b()
    tae_path = Path(a.tae) if a.tae else next(iter(model_dir.glob("*tae*.safetensors")), None)
    if tae_path is None:
        _log("no TAEHV file: pass --tae taew2_1.safetensors")
        return 2
    tcfg = TaeConfig.from_file_name(tae_path.name)
    sizes = [Geometry.parse(s) for s in a.size] if a.size else [FASTWAN_480P, MOBILE_480x272]
    export(
        Path(a.out), cfg, tcfg, SafetensorsDir(tdir), SafetensorsDir(tae_path), src_spec=("safetensors", str(tdir)), model_name="fastwan-1.3b",
        weights=a.weights, chunk=a.chunk, q_chunk=a.q_chunk, sizes=sizes, tae_frames=a.tae_frames, jobs=a.jobs, log=_log,
    )
    return 0


def _open(a: argparse.Namespace):
    from .pipeline import Pipeline
    from .runtime import GraphStore, Manifest

    m = Manifest(a.graphs)
    return Pipeline(GraphStore(m, accelerator=a.accelerator, resident=a.resident, threads=a.threads))


def _load_embeds(path: str, cfg: WanConfig) -> np.ndarray:
    from safetensors.numpy import load_file

    t = load_file(path)
    key = "prompt_embeds" if "prompt_embeds" in t else next(iter(t))
    e = t[key].astype(np.float32)
    if e.shape != (cfg.text_len, cfg.text_dim):
        raise SystemExit(f"{path}: {key} is {e.shape}, expected {(cfg.text_len, cfg.text_dim)}")
    return e


def _write_video(path: Path, video, fps: int = 16) -> None:
    """`.y4m` (ffmpeg reads it; no codec dependency) or `.png` of the first frame."""
    if path.suffix == ".png":
        from .media import write_png

        write_png(path, video.rgb[0])
        return
    from .media import write_y4m

    write_y4m(path, video.rgb, fps)


def cmd_generate(a: argparse.Namespace) -> int:
    from . import plan as planmod

    pipe = _open(a)
    g = Geometry.parse(a.size)
    g.validate(pipe.cfg)
    p = planmod.plan(pipe.cfg, pipe.m.tae, g, pipe.m.weights, pipe.m.chunk, pipe.m.attention_for(g.tokens(pipe.cfg))["q_chunk"], pipe.m.tae_for(*g.latent()[1:])["frames"], int(a.budget_gib * (1 << 30)))
    _log(str(p))
    if not p.fits and not a.force:
        _log("the plan does not fit the budget; pass --force or a larger --budget-gib")
        return 3
    embeds = _load_embeds(a.embeds, pipe.cfg)

    def on(ev):
        if ev["type"] == "step":
            _log(f"step {ev['index'] + 1}/{ev['total']}: {ev['ms'] / 1000:.1f} s")
        elif ev["type"] == "frames":
            _log(f"frames {ev['first']}..{ev['first'] + ev['count']} {ev['width']}x{ev['height']}")
        elif ev["type"] == "prepared":
            _log(f"prepared {ev['tokens']} tokens in {ev['ms'] / 1000:.1f} s")

    video, tm = pipe.generate(embeds, g, a.seed, on)
    _write_video(Path(a.out), video)
    _log(f"{a.out}: {video.frames} frames {video.width}x{video.height}; denoise {sum(tm.steps_s):.1f} s, decode {tm.decode_s:.1f} s")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    """Run the tiny fixture through the graph pipeline and compare with `expected.safetensors`."""
    from safetensors.numpy import load_file

    pipe = _open(a)
    exp = load_file(str(Path(a.graphs) / "expected.safetensors"))
    size = pipe.m.raw["sizes"][0]
    g = Geometry(size["width"], size["height"], size["frames"])
    seed = int(_meta(Path(a.graphs) / "expected.safetensors").get("seed", "7"))
    t0 = time.time()
    latent = pipe.denoise(exp["prompt_embeds"], g, seed)
    rel = float(np.abs(latent - exp["latent"]).max() / np.abs(exp["latent"]).max())
    video = pipe.decode(latent)
    fdiff = int(np.abs(video.rgb.astype(np.int32) - exp["frames"].astype(np.int32)).max())
    ok = rel <= a.tol and fdiff <= 2
    _log(f"latent rel {rel:.2e} (tol {a.tol:.0e}), frames max |Δ| {fdiff}/255 in {time.time() - t0:.1f} s: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def _meta(path: Path) -> dict:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {}) or {}


def cmd_plan(a: argparse.Namespace) -> int:
    from . import plan as planmod

    cfg = WanConfig.from_name(a.model)
    tcfg = TaeConfig.tiny() if a.model == "tiny" else TaeConfig.taew2_1()
    g = Geometry.parse(a.size)
    g.validate(cfg)
    s = g.tokens(cfg)
    p = planmod.plan(cfg, tcfg, g, a.weights, a.chunk, min(a.q_chunk, s), a.tae_frames, int(a.budget_gib * (1 << 30)), a.resident)
    print(p)
    return 0 if p.fits else 3


def cmd_encode(a: argparse.Namespace) -> int:
    from .encode import encode_prompt

    encode_prompt(a.model_dir, a.prompt, Path(a.out), log=_log)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    pipe = _open(a)
    serve(pipe, a.host, a.port, Path(a.gallery) if a.gallery else None, log=_log)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fastwan-litert", description="FastWan 2.1 1.3B on LiteRT")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="checkpoint → .tflite graphs + manifest")
    e.add_argument("--model", default="fastwan-1.3b", choices=["fastwan-1.3b", "tiny"])
    e.add_argument("--model-dir", help="Diffusers download (transformer/ inside, or the transformer dir itself)")
    e.add_argument("--tae", help="taew2_1.safetensors (default: *tae*.safetensors in --model-dir)")
    e.add_argument("--out", required=True)
    e.add_argument("--weights", default="f16", choices=["f32", "f16", "int8", "int4"])
    e.add_argument("--chunk", type=int, default=1024, help="rows per per-row graph call")
    e.add_argument("--q-chunk", type=int, default=256, help="queries per attention call")
    e.add_argument("--tae-frames", type=int, default=3, help="latent frames per TAEHV chunk")
    e.add_argument("--size", action="append", help="WxHxF to export attention / TAEHV graphs for (repeatable; default 832x480x81 and 480x272x33)")
    e.add_argument("--jobs", type=int, default=1, help="block graphs converted in parallel worker processes")
    e.set_defaults(fn=cmd_export)

    def runtime_args(p):
        p.add_argument("--graphs", required=True, help="the export directory (manifest.json)")
        p.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
        p.add_argument("--resident", type=int, default=4, help="block graphs kept open at once")
        p.add_argument("--threads", type=int)

    gen = sub.add_parser("generate", help="prompt embeddings → clip")
    runtime_args(gen)
    gen.add_argument("--embeds", required=True, help="safetensors with prompt_embeds [512, 4096] (fastwan-litert encode)")
    gen.add_argument("--size", default="832x480x81")
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--out", default="out.y4m", help=".y4m (ffmpeg -i out.y4m out.mp4) or .png of the first frame")
    gen.add_argument("--budget-gib", type=float, default=16.0)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(fn=cmd_generate)

    ch = sub.add_parser("check", help="run the tiny fixture and compare with expected.safetensors")
    runtime_args(ch)
    ch.add_argument("--tol", type=float, default=1e-3)
    ch.set_defaults(fn=cmd_check)

    pl = sub.add_parser("plan", help="memory plan for a size and weight format")
    pl.add_argument("--model", default="fastwan-1.3b")
    pl.add_argument("--size", default="832x480x81")
    pl.add_argument("--weights", default="f16", choices=["f32", "f16", "int8", "int4"])
    pl.add_argument("--chunk", type=int, default=1024)
    pl.add_argument("--q-chunk", type=int, default=256)
    pl.add_argument("--tae-frames", type=int, default=3)
    pl.add_argument("--resident", type=int, default=2)
    pl.add_argument("--budget-gib", type=float, default=16.0)
    pl.set_defaults(fn=cmd_plan)

    en = sub.add_parser("encode", help="UMT5-XXL prompt embeddings (needs torch + transformers, 11 GB of weights)")
    en.add_argument("--model-dir", required=True, help="the FastWan download (text_encoder/, tokenizer/)")
    en.add_argument("--prompt", required=True)
    en.add_argument("--out", required=True)
    en.set_defaults(fn=cmd_encode)

    sv = sub.add_parser("serve", help="local studio: the demo page on this machine's graphs")
    runtime_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7860)
    sv.add_argument("--gallery", help="directory of <name>.safetensors prompt embeddings")
    sv.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
