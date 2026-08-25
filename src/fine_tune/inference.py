from typing import cast

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)

from src.fine_tune.response import parse_response

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
print(generated_text)

for text in generated_text:
    # print(tokenizer.parse_response(text))
    print(parse_response(text))
