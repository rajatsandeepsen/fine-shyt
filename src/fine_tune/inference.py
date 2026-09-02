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

from app.api.chat.request import Message
from app.api.chat.response import AssistantMessage

from .config import Config

base = Config.model_name
adapter_dir = "fine-tuned-model"

tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained(adapter_dir))
base_model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(base_model, adapter_dir)

if not tokenizer.response_schema:
    parent_path = Path(__file__).resolve().parent
    schema_path = parent_path / "schema.json"
    response_schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))

    tokenizer.response_schema = response_schema


def Inference(messages: list[Message] | list[dict]) -> list[AssistantMessage]:
    input = tokenizer.apply_chat_template(
        conversation=cast(list[dict], messages),
        tools=[],
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

    return cast(
        list[AssistantMessage],
        [tokenizer.parse_response(text) for text in generated_text],
    )


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Who built you?"},
    ]

    print(Inference(messages))
