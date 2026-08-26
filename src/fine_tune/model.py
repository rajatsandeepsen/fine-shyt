import json
from pathlib import Path
from typing import cast

from datasets import Dataset
from jinja2 import Template
from pydantic import BaseModel
from unsloth import FastModel, train_on_responses_only

# isort: off
# always import trl after unsloth
from trl import SFTConfig, SFTTrainer

from app.api.chat.request import Message


class Data(BaseModel):
    messages: list[Message]


data: list[Data] = [
    Data(
        messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hey. What do you want?"),
        ]
    )
]
# print(data[0].messages[0].content)

raw_data: list[dict] = [item.model_dump(mode="json") for item in data]
# print(raw_data[0]["messages"][0]["content"])

parent_path = Path(__file__).resolve().parent

template_path = parent_path / "template.jinja"
chat_template = template_path.read_text(encoding="utf-8")
template = Template(chat_template)

schema_path = parent_path / "schema.json"
response_schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))


def to_text(data: dict) -> dict:
    text = cast(str, template.render(**data)).strip()
    print(text)
    return {"text": text}


train_dataset: Dataset = Dataset.from_list(raw_data).map(
    to_text, remove_columns=["messages"]
)

print(train_dataset)


source_model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/SmolLM2-360M-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model=source_model,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

tokenizer.chat_template = chat_template
tokenizer.response_schema = response_schema

if not tokenizer.chat_template:
    raise ValueError("tokenizer.chat_template is empty or None")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=30,
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer_stats = trainer.train()
print(trainer_stats)

# if we want merged model, not just adapter weights
if False:
    model = model.merge_and_unload()

model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")
