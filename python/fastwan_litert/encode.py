"""UMT5-XXL prompt embeddings, as Diffusers' `WanPipeline._get_t5_prompt_embeds`
makes them: 512 tokens, the rows past the prompt zeroed. The same file
fastwan-rs's `scripts/encode_prompt.py` writes (`prompt_embeds [512, 4096]`)."""

from __future__ import annotations

from pathlib import Path


def clean_prompt(p: str) -> str:
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        p = p.replace(a, b)
    return " ".join(p.split())


def encode_prompt(model_dir: str, prompt: str, out: Path, text_len: int = 512, log=print) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(dev).eval()
    t = tok([clean_prompt(prompt)], padding="max_length", max_length=text_len, truncation=True, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        h = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0].float()
    n = int(t.attention_mask.sum())
    h[n:] = 0
    save_file({"prompt_embeds": h.cpu().contiguous()}, str(out), metadata={"prompt": prompt, "tokens": str(n)})
    log(f"{out}: [{text_len}, {h.shape[1]}], {n} prompt tokens")
