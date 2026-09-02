from __future__ import annotations

from pathlib import Path

from unsloth import FastModel

from src.fine_tune.config import Config

model, tokenizer = FastModel.from_pretrained(**Config.as_dict())

template = tokenizer.chat_template

if not template:
    raise ValueError("tokenizer.chat_template is empty or None")

out_path = Path(__file__).resolve().parent / "template.jinja"
Path(out_path).write_text(template, encoding="utf-8")
