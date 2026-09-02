from typing import Any


class Config:
    model_name: str = "unsloth/SmolLM2-360M-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {key: getattr(cls, key) for key in cls.__annotations__}
