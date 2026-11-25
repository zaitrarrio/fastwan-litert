"""The PyTorch DiT (modules.WanDit, built from the graph modules) against
Diffusers' `WanTransformer3DModel`: one forward on tiny random weights, the
same fixture recipe as fastwan-rs's `scripts/reference/make_fixtures.py`."""

import numpy as np
import pytest
import torch

from fastwan_litert.config import WanConfig
from fastwan_litert.modules import WanDit
from fastwan_litert.rope import rope_table
from fastwan_litert.weights import load_dit

diffusers = pytest.importorskip("diffusers")


class DictSource:
    def __init__(self, sd):
        self.sd = {k: v.detach().float().numpy() for k, v in sd.items()}

    def contains(self, name):
        return name in self.sd

    def get(self, name, shape):
        t = self.sd[name]
        assert tuple(d for d in t.shape if d != 1) == tuple(d for d in shape if d != 1), (name, t.shape, shape)
        return t.reshape(shape)


def jitter(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))


def test_wan_dit_matches_diffusers():
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    cfg = WanConfig.tiny()
    ref = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=cfg.heads, attention_head_dim=cfg.head_dim,
        in_channels=cfg.in_channels, out_channels=cfg.out_channels, text_dim=cfg.text_dim,
        freq_dim=cfg.freq_dim, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=cfg.eps, rope_max_seq_len=cfg.rope_max_seq_len,
    ).eval()
    jitter(ref)
    t, h, w = 3, 8, 12
    x = torch.randn(1, cfg.in_channels, t, h, w)
    ctx = torch.randn(1, cfg.text_len, cfg.text_dim)
    ts = torch.tensor([757.0])
    with torch.no_grad():
        want = ref(hidden_states=x, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0][0]
    dit = WanDit(cfg).eval()
    load_dit(dit, DictSource(ref.state_dict()))
    cos, sin = rope_table(cfg, (t, h // cfg.patch, w // cfg.patch), 0)
    with torch.no_grad():
        got = dit(x[0], ctx[0], ts, torch.from_numpy(cos), torch.from_numpy(sin))
    scale = want.abs().max().item()
    err = (got - want).abs().max().item() / scale
    print(f"Wan DiT vs diffusers: rel {err:e}")
    assert err < 1e-4
