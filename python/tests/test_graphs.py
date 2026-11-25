"""The graph cut against the uncut PyTorch model, through `ai-edge-litert`:
the tiny export's graphs run by the pipeline equal the reference denoise and
decode; and a quantized export stays close."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

litert_torch = pytest.importorskip("litert_torch")

from fastwan_litert.export import TINY_GEOMETRY, TINY_SEED, export_tiny  # noqa: E402
from fastwan_litert.pipeline import Pipeline  # noqa: E402
from fastwan_litert.runtime import GraphStore, Manifest  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    export_tiny(d, weights="f32", log=lambda *a: None)
    return d


def test_manifest_is_the_contract(tiny):
    m = json.loads((tiny / "manifest.json").read_text())
    assert m["format"] == "fastwan-litert/1"
    assert m["chunk"] == 16 and len(m["graphs"]["blocks"]) == 2
    assert m["attention"][0] == {"tokens": 18, "q_chunk": 8, "heads": 2, "file": "attn_s18_q8_h2.tflite"}
    assert m["tae_chunks"][0]["file"] == "tae_f2_6x4.tflite"
    for f, info in m["files"].items():
        assert (tiny / f).stat().st_size == info["bytes"]


def test_graph_pipeline_matches_reference(tiny):
    exp = load_file(str(tiny / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    events = []
    latent = pipe.denoise(exp["prompt_embeds"], TINY_GEOMETRY, TINY_SEED, events.append)
    rel = np.abs(latent - exp["latent"]).max() / np.abs(exp["latent"]).max()
    assert rel < 1e-4, rel
    video = pipe.decode(latent, events.append)
    assert video.rgb.shape == exp["frames"].shape == (9, 48, 32, 3)
    assert np.abs(video.rgb.astype(int) - exp["frames"].astype(int)).max() <= 2
    kinds = [e["type"] for e in events]
    assert kinds[:4] == ["prepared", "step", "step", "step"] and kinds[4:] == ["frames", "frames"]
    assert sum(e["count"] for e in events if e["type"] == "frames") == 9


def test_seed_changes_the_clip(tiny):
    exp = load_file(str(tiny / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    other = pipe.denoise(exp["prompt_embeds"], TINY_GEOMETRY, TINY_SEED + 1)
    assert np.abs(other - exp["latent"]).max() > 0.1


def test_int8_export_stays_close(tmp_path):
    exp_dir = Path(tmp_path) / "q8"
    export_tiny(exp_dir, weights="int8", log=lambda *a: None)
    m = json.loads((exp_dir / "manifest.json").read_text())
    assert m["weights"] == "int8"
    exp = load_file(str(exp_dir / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(exp_dir)))
    latent = pipe.denoise(exp["prompt_embeds"], TINY_GEOMETRY, TINY_SEED)
    rel = np.abs(latent - exp["latent"]).max() / np.abs(exp["latent"]).max()
    # Tiny random weights quantize badly (fastwan-rs sees int8 within the backend gap on the real model);
    # the point here is that the quantized graphs load, run and stay in the right ballpark.
    assert np.isfinite(latent).all() and rel < 0.5, rel
