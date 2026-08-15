from __future__ import annotations

from pathlib import Path

from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/SmolLM2-360M-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

template = tokenizer.chat_template

if not template:
    raise ValueError("tokenizer.chat_template is empty or None")

out_path = Path(__file__).resolve().parent / "original.jinja"
Path(out_path).write_text(template, encoding="utf-8")
