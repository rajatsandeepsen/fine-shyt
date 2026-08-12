from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/SmolLM2-360M",
    max_seq_length=2048,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    # token="",
)
