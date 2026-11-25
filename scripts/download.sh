#!/usr/bin/env bash
# The FastWan 2.1 1.3B checkpoint, TAEHV and the text encoder into ./fastwan.
set -euo pipefail
dir="${1:-fastwan}"
huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --include "transformer/*" "text_encoder/*" "tokenizer/*" --local-dir "$dir"
huggingface-cli download madebyollin/taehv taew2_1.safetensors --local-dir "$dir"
ls -la "$dir"
