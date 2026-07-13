#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under the MIT License.
#
# Helper for the Phase-1 e2e test. Two modes:
#   encode <prompt> <out.txt>        Tokenize a chat prompt -> whitespace-separated ids.
#   decode <ids...>                  Decode generated token ids back to text (coherence read-out).
#
# Uses the tokenizer.json shipped with the model package (Qwen2.5 byte-level BPE).

import sys
from pathlib import Path

MODEL_DIR = Path("/Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-cpu-recipe")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def chat_prompt(user_msg: str) -> str:
    # Qwen2.5 ChatML template.
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    if len(sys.argv) < 2:
        print("usage: prepare_prompt.py encode <prompt> <out.txt> | decode <ids...>", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    tok = load_tokenizer()
    if mode == "encode":
        prompt = sys.argv[2]
        out = sys.argv[3]
        ids = tok.encode(chat_prompt(prompt)).ids
        Path(out).write_text(" ".join(str(i) for i in ids))
        print(f"wrote {len(ids)} tokens to {out}")
        print("ids:", ids)
    elif mode == "decode":
        ids = [int(x) for x in sys.argv[2:]]
        print(tok.decode(ids))
    else:
        print(f"unknown mode {mode}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
