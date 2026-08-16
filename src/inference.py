from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from unsloth import FastModel

MODEL_NAME = "unsloth/SmolLM2-360M-Instruct"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
FastModel.for_inference(model)

app = FastAPI(title="Inference API", version="1.0.0")


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Input prompt")
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    do_sample: bool = Field(default=True)


class InferenceResponse(BaseModel):
    text: str
    model_name: str
    input_tokens: int
    output_tokens: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model is not None and tokenizer is not None,
        model_name=MODEL_NAME,
    )


@app.post("/infer", response_model=InferenceResponse)
def infer(payload: InferenceRequest) -> InferenceResponse:
    try:
        inputs = tokenizer(payload.prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=payload.max_new_tokens,
                do_sample=payload.do_sample,
                temperature=payload.temperature,
                top_p=payload.top_p,
                repetition_penalty=payload.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = int(inputs["input_ids"].shape[-1])
        new_token_ids = output_ids[0][prompt_len:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        return InferenceResponse(
            text=text,
            model_name=MODEL_NAME,
            input_tokens=prompt_len,
            output_tokens=int(new_token_ids.shape[-1]),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        ) from exc


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Inference server is running",
        "endpoints": ["GET /health", "POST /infer"],
    }
