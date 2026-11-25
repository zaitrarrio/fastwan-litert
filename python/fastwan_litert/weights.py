"""Checkpoint access: a Diffusers safetensors tree (any dtype, any number of
shards) or deterministic synthetic weights of the same names and shapes, and
the mapping of those names onto the graph modules' parameters.

The model asks for every tensor by its exact Diffusers key and shape; a
missing key or a shape mismatch is an error, never a default (fastwan-rs
learned that a silently-defaulted weight streams gray haze)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .config import TaeConfig, WanConfig
from .rng import Rng


class WeightSource(Protocol):
    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray: ...
    def contains(self, name: str) -> bool: ...


def _squeeze(s) -> tuple[int, ...]:
    return tuple(d for d in s if d != 1)


class SafetensorsDir:
    """All `*.safetensors` under a directory (or one file), read lazily."""

    def __init__(self, path: Path | str):
        from safetensors import safe_open

        path = Path(path)
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .safetensors in {path}")
        # torch framework: bf16 checkpoints read where numpy's reader refuses them
        self._files = [safe_open(str(f), framework="pt") for f in files]
        self._index: dict[str, int] = {}
        for i, f in enumerate(self._files):
            for k in f.keys():
                self._index[k] = i

    def contains(self, name: str) -> bool:
        return name in self._index

    def names(self):
        return self._index.keys()

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self._index:
            raise KeyError(f"missing tensor `{name}`")
        import torch

        t = self._files[self._index[name]].get_tensor(name).to(torch.float32).numpy()
        if _squeeze(t.shape) != _squeeze(shape):
            raise ValueError(f"`{name}` is {list(t.shape)}, expected {list(shape)}")
        return t.reshape(shape)



class Synthetic:
    """Deterministic uniform weights by name hash: benchmarks and the tiny
    fixture without a download. Norm weights sit near 1, biases near 0, and
    linears scale with fan-in so activations stay O(1) through 30 blocks."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def contains(self, name: str) -> bool:
        return True

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(shape))
        h = 1469598103934665603
        for ch in name.encode():
            h = ((h ^ ch) * 1099511628211) & ((1 << 64) - 1)
        rng = Rng(self.seed ^ h)
        if name.endswith("norm_q.weight") or name.endswith("norm_k.weight") or name.endswith("norm2.weight"):
            return (1.0 + rng.uniform(n, 0.1)).reshape(shape)
        if name.endswith(".bias") or name.endswith("scale_shift_table"):
            return rng.uniform(n, 0.1).reshape(shape)
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        return rng.uniform(n, float(np.sqrt(3.0 / fan_in))).reshape(shape)


# ---- manifests ----------------------------------------------------------------


def wan_manifest(cfg: WanConfig) -> list[tuple[str, tuple[int, ...]]]:
    d = cfg.dim
    v: list[tuple[str, tuple[int, ...]]] = [
        ("patch_embedding.weight", (d, cfg.in_channels, 1, cfg.patch, cfg.patch)),
        ("patch_embedding.bias", (d,)),
        ("condition_embedder.time_embedder.linear_1.weight", (d, cfg.freq_dim)),
        ("condition_embedder.time_embedder.linear_1.bias", (d,)),
        ("condition_embedder.time_embedder.linear_2.weight", (d, d)),
        ("condition_embedder.time_embedder.linear_2.bias", (d,)),
        ("condition_embedder.time_proj.weight", (6 * d, d)),
        ("condition_embedder.time_proj.bias", (6 * d,)),
        ("condition_embedder.text_embedder.linear_1.weight", (d, cfg.text_dim)),
        ("condition_embedder.text_embedder.linear_1.bias", (d,)),
        ("condition_embedder.text_embedder.linear_2.weight", (d, d)),
        ("condition_embedder.text_embedder.linear_2.bias", (d,)),
        ("scale_shift_table", (1, 2, d)),
        ("proj_out.weight", (cfg.patch_features_out(), d)),
        ("proj_out.bias", (cfg.patch_features_out(),)),
    ]
    for l in range(cfg.layers):
        p = f"blocks.{l}"
        for a in ("attn1", "attn2"):
            for m in ("to_q", "to_k", "to_v", "to_out.0"):
                v.append((f"{p}.{a}.{m}.weight", (d, d)))
                v.append((f"{p}.{a}.{m}.bias", (d,)))
            v.append((f"{p}.{a}.norm_q.weight", (d,)))
            v.append((f"{p}.{a}.norm_k.weight", (d,)))
        v.append((f"{p}.norm2.weight", (d,)))
        v.append((f"{p}.norm2.bias", (d,)))
        v.append((f"{p}.ffn.net.0.proj.weight", (cfg.ffn_dim, d)))
        v.append((f"{p}.ffn.net.0.proj.bias", (cfg.ffn_dim,)))
        v.append((f"{p}.ffn.net.2.weight", (d, cfg.ffn_dim)))
        v.append((f"{p}.ffn.net.2.bias", (d,)))
        v.append((f"{p}.scale_shift_table", (1, 6, d)))
    return v


def tae_manifest(cfg: TaeConfig) -> list[tuple[str, tuple[int, ...]]]:
    nf = cfg.widths
    v: list[tuple[str, tuple[int, ...]]] = [
        ("decoder.1.weight", (nf[0], cfg.latent_channels, 3, 3)),
        ("decoder.1.bias", (nf[0],)),
    ]
    for s in range(3):
        base = 3 + 6 * s
        for m in range(cfg.mem_blocks):
            p = f"decoder.{base + m}"
            v += [
                (f"{p}.conv.0.weight", (nf[s], 2 * nf[s], 3, 3)),
                (f"{p}.conv.0.bias", (nf[s],)),
                (f"{p}.conv.2.weight", (nf[s], nf[s], 3, 3)),
                (f"{p}.conv.2.bias", (nf[s],)),
                (f"{p}.conv.4.weight", (nf[s], nf[s], 3, 3)),
                (f"{p}.conv.4.bias", (nf[s],)),
            ]
        v.append((f"decoder.{base + 4}.conv.weight", (nf[s] * cfg.tgrow[s], nf[s], 1, 1)))
        v.append((f"decoder.{base + 5}.weight", (nf[s + 1], nf[s], 3, 3)))
    v += [
        ("decoder.22.weight", (cfg.out_channels(), nf[3], 3, 3)),
        ("decoder.22.bias", (cfg.out_channels(),)),
    ]
    return v


# ---- loading into the modules ---------------------------------------------------


def _t(src: WeightSource, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(src.get(name, shape)))


@torch.no_grad()
def load_block(l: int, qkv, kv, post, src: WeightSource, cfg: WanConfig) -> None:
    """Block `l`'s tensors into its three graph modules (`modules.BlockQkv`, `BlockKV`, `BlockPost`)."""
    d = cfg.dim
    p = f"blocks.{l}"

    def lin(mod: torch.nn.Linear, key: str, shape: tuple[int, int]):
        mod.weight.copy_(_t(src, f"{key}.weight", shape))
        mod.bias.copy_(_t(src, f"{key}.bias", (shape[0],)))

    lin(qkv.to_q, f"{p}.attn1.to_q", (d, d))
    lin(qkv.to_k, f"{p}.attn1.to_k", (d, d))
    lin(qkv.to_v, f"{p}.attn1.to_v", (d, d))
    qkv.norm_q.copy_(_t(src, f"{p}.attn1.norm_q.weight", (d,)))
    qkv.norm_k.copy_(_t(src, f"{p}.attn1.norm_k.weight", (d,)))
    lin(post.to_out, f"{p}.attn1.to_out.0", (d, d))
    lin(kv.to_k, f"{p}.attn2.to_k", (d, d))
    lin(kv.to_v, f"{p}.attn2.to_v", (d, d))
    kv.norm_k.copy_(_t(src, f"{p}.attn2.norm_k.weight", (d,)))
    lin(post.to_q2, f"{p}.attn2.to_q", (d, d))
    post.norm_q2.copy_(_t(src, f"{p}.attn2.norm_q.weight", (d,)))
    lin(post.to_out2, f"{p}.attn2.to_out.0", (d, d))
    post.norm2.weight.copy_(_t(src, f"{p}.norm2.weight", (d,)))
    post.norm2.bias.copy_(_t(src, f"{p}.norm2.bias", (d,)))
    lin(post.ffn_in, f"{p}.ffn.net.0.proj", (cfg.ffn_dim, d))
    lin(post.ffn_out, f"{p}.ffn.net.2", (d, cfg.ffn_dim))


@torch.no_grad()
def load_dit(dit, src: WeightSource) -> None:
    """Fill a `modules.WanDit` from Diffusers-named tensors."""
    cfg: WanConfig = dit.cfg
    d = cfg.dim
    ce = "condition_embedder"
    tm = dit.time
    tm.linear_1.weight.copy_(_t(src, f"{ce}.time_embedder.linear_1.weight", (d, cfg.freq_dim)))
    tm.linear_1.bias.copy_(_t(src, f"{ce}.time_embedder.linear_1.bias", (d,)))
    tm.linear_2.weight.copy_(_t(src, f"{ce}.time_embedder.linear_2.weight", (d, d)))
    tm.linear_2.bias.copy_(_t(src, f"{ce}.time_embedder.linear_2.bias", (d,)))
    tm.time_proj.weight.copy_(_t(src, f"{ce}.time_proj.weight", (6 * d, d)))
    tm.time_proj.bias.copy_(_t(src, f"{ce}.time_proj.bias", (6 * d,)))
    tm.head_table.copy_(_t(src, "scale_shift_table", (1, 2, d)).reshape(2, d))
    dit.text.linear_1.weight.copy_(_t(src, f"{ce}.text_embedder.linear_1.weight", (d, cfg.text_dim)))
    dit.text.linear_1.bias.copy_(_t(src, f"{ce}.text_embedder.linear_1.bias", (d,)))
    dit.text.linear_2.weight.copy_(_t(src, f"{ce}.text_embedder.linear_2.weight", (d, d)))
    dit.text.linear_2.bias.copy_(_t(src, f"{ce}.text_embedder.linear_2.bias", (d,)))
    dit.patch.proj.weight.copy_(
        _t(src, "patch_embedding.weight", (d, cfg.in_channels, 1, cfg.patch, cfg.patch)).reshape(d, -1)
    )
    dit.patch.proj.bias.copy_(_t(src, "patch_embedding.bias", (d,)))
    dit.head.proj_out.weight.copy_(_t(src, "proj_out.weight", (cfg.patch_features_out(), d)))
    dit.head.proj_out.bias.copy_(_t(src, "proj_out.bias", (cfg.patch_features_out(),)))
    for l in range(cfg.layers):
        tm.block_tables[l].copy_(_t(src, f"blocks.{l}.scale_shift_table", (1, 6, d)).reshape(6, d))
        if l < len(dit.qkv):
            load_block(l, dit.qkv[l], dit.kv[l], dit.post[l], src, cfg)


def open_source(spec):
    """`("safetensors", path)` or `("synthetic", seed)` → a WeightSource (picklable for worker processes)."""
    kind, arg = spec
    return SafetensorsDir(arg) if kind == "safetensors" else Synthetic(int(arg))


@torch.no_grad()
def load_tae(tae, src: WeightSource) -> None:
    cfg: TaeConfig = tae.cfg
    nf = cfg.widths

    def conv(mod: torch.nn.Conv2d, key: str, cout: int, cin: int, k: int, bias: bool):
        mod.weight.copy_(_t(src, f"{key}.weight", (cout, cin, k, k)))
        if bias:
            mod.bias.copy_(_t(src, f"{key}.bias", (cout,)))

    conv(tae.first, "decoder.1", nf[0], cfg.latent_channels, 3, True)
    for s, stage in enumerate(tae.stages):
        base = 3 + 6 * s
        for m, mem in enumerate(stage.mems):
            p = f"decoder.{base + m}"
            conv(mem.conv0, f"{p}.conv.0", nf[s], 2 * nf[s], 3, True)
            conv(mem.conv2, f"{p}.conv.2", nf[s], nf[s], 3, True)
            conv(mem.conv4, f"{p}.conv.4", nf[s], nf[s], 3, True)
        conv(stage.grow, f"decoder.{base + 4}.conv", nf[s] * cfg.tgrow[s], nf[s], 1, False)
        conv(stage.conv, f"decoder.{base + 5}", nf[s + 1], nf[s], 3, False)
    conv(tae.last, "decoder.22", cfg.out_channels(), nf[3], 3, True)


def read_wan_config(model_dir: Path) -> WanConfig:
    """`transformer/config.json` of a Diffusers download."""
    cj = json.loads((Path(model_dir) / "config.json").read_text())
    return WanConfig(
        patch=cj["patch_size"][1],
        heads=cj["num_attention_heads"],
        head_dim=cj["attention_head_dim"],
        in_channels=cj["in_channels"],
        out_channels=cj["out_channels"],
        text_dim=cj["text_dim"],
        text_len=512,
        freq_dim=cj["freq_dim"],
        ffn_dim=cj["ffn_dim"],
        layers=cj["num_layers"],
        eps=cj.get("eps", 1e-6),
        rope_max_seq_len=cj.get("rope_max_seq_len", 1024),
        rope_theta=10000.0,
    )
