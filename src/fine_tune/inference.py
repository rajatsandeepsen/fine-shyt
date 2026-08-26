import json
from pathlib import Path
from typing import cast

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)

base = "unsloth/SmolLM2-360M-Instruct"
adapter_dir = "fine-tuned-model"

tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained(adapter_dir))
base_model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(base_model, adapter_dir)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who built you?"},
]

input = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

input = cast(BatchEncoding, input).to(model.device)
print(input)

output = model.generate(
    **input,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

print(output)

generated_text = tokenizer.batch_decode(
    output,
    skip_special_tokens=True,
)

parent_path = Path(__file__).resolve().parent
schema_path = parent_path / "schema.json"
response_schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))

tokenizer.response_schema = response_schema

for text in generated_text:
    print(tokenizer.parse_response(text))
