"""Graph pipeline against the uncut PyTorch model at real weights: one clip's
denoise (latent) and decode (frames), at a small size. `docs/runs/` records
the numbers. Needs the Diffusers download, the TAEHV file and an export.

    python scripts/parity_real.py --model-dir /root/work/fastwan --tae taew2_1.safetensors \
        --graphs graphs/fastwan-f16 --embeds gallery/fox.safetensors --size 480x272x33
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fastwan_litert import reference
from fastwan_litert.config import TaeConfig
from fastwan_litert.geometry import Geometry
from fastwan_litert.modules import TaeHv, WanDit
from fastwan_litert.pipeline import Pipeline
from fastwan_litert.runtime import GraphStore, Manifest
from fastwan_litert.weights import SafetensorsDir, load_dit, load_tae, read_wan_config


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tae", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--size", default="480x272x33")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--out", help="json report")
    a = ap.parse_args()
    from safetensors.numpy import load_file

    w, h, f = (int(v) for v in a.size.split("x"))
    g = Geometry(w, h, f)
    embeds = load_file(a.embeds)["prompt_embeds"].astype(np.float32)
    model_dir = Path(a.model_dir)
    tdir = model_dir / "transformer" if (model_dir / "transformer").is_dir() else model_dir
    cfg = read_wan_config(tdir)
    torch.set_num_threads(torch.get_num_threads())

    m = Manifest(a.graphs)
    pipe = Pipeline(GraphStore(m, accelerator=a.accelerator))
    t0 = time.time()
    lat_g = pipe.denoise(embeds, g, a.seed)
    t_den_g = time.time() - t0
    t0 = time.time()
    vid_g = pipe.decode(lat_g)
    t_dec_g = time.time() - t0
    print(f"graphs: denoise {t_den_g:.1f} s, decode {t_dec_g:.1f} s", flush=True)

    dit = WanDit(cfg).eval()
    load_dit(dit, SafetensorsDir(tdir))
    tcfg = TaeConfig.from_file_name(Path(a.tae).name)
    tae = TaeHv(tcfg).eval()
    load_tae(tae, SafetensorsDir(a.tae))
    t0 = time.time()
    lat_r = reference.denoise(dit, m.schedule, embeds, g, a.seed)
    t_den_r = time.time() - t0
    t0 = time.time()
    frames_r = reference.decode(tae, lat_r, int(m.tae_for(*lat_r.shape[2:])["frames"]))
    t_dec_r = time.time() - t0
    frames_g_of_r = pipe.decode(lat_r).rgb  # the TAE graph on the reference latent: decode-only error
    print(f"torch:  denoise {t_den_r:.1f} s, decode {t_dec_r:.1f} s", flush=True)

    # per-forward error on the reference trajectory (no divergence folded in)
    from fastwan_litert.host import dmd_step
    from fastwan_litert.rng import Rng
    from fastwan_litert.rope import rope_table

    cos, sin = rope_table(cfg, g.grid(cfg), 0)
    attn = m.attention_for(g.tokens(cfg))
    prompt = pipe.prepare_prompt(embeds)
    lt, lh, lw = g.latent()
    rng = Rng(a.seed)
    x = rng.normal(cfg.in_channels * lt * lh * lw).reshape(cfg.in_channels, lt, lh, lw)
    emb_t = torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32))
    cos_t, sin_t = torch.from_numpy(cos), torch.from_numpy(sin)
    step_err = []
    with torch.no_grad():
        for i, t in enumerate(m.schedule.timesteps):
            v_r = dit(torch.from_numpy(x), emb_t, torch.tensor([float(t)]), cos_t, sin_t).numpy()
            v_g = pipe.forward(x, prompt, float(t), cos, sin, attn)
            step_err.append(rel(v_g, v_r))
            sigma, nxt = m.schedule.coeffs(i)
            noise = rng.normal(x.size).reshape(x.shape) if nxt is not None else None
            x = dmd_step(x, v_r, sigma, nxt, noise)
    print("per-step velocity rel err (graphs vs torch, same input):", [f"{e:.4f}" for e in step_err], flush=True)

    d_lat = rel(lat_g, lat_r)
    d_dec = int(np.abs(frames_g_of_r.astype(np.int16) - frames_r.astype(np.int16)).max())
    d_dec_mean = float(np.abs(frames_g_of_r.astype(np.int16) - frames_r.astype(np.int16)).mean())
    d_all = float(np.abs(vid_g.rgb.astype(np.int16) - frames_r.astype(np.int16)).mean())
    rep = {
        "size": a.size, "seed": a.seed, "weights": m.raw.get("weights"), "accelerator": a.accelerator,
        "latent_rel_err": d_lat, "step_velocity_rel_err": step_err, "decode_max_abs_8bit": d_dec, "decode_mean_abs_8bit": d_dec_mean,
        "end_to_end_mean_abs_8bit": d_all,
        "graphs_denoise_s": t_den_g, "graphs_decode_s": t_dec_g, "torch_denoise_s": t_den_r, "torch_decode_s": t_dec_r,
        "latent_shape": list(lat_r.shape), "frames": list(frames_r.shape),
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
