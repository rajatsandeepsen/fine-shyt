from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

api = APIRouter()

FinishReason = Literal[
    "stop",
    "length",
    "content_filter",
    "function_call",
    "tool_calls",
    "error",
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Request(BaseModel):
    model: str
    messages: list[Message]

    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    max_tokens: int | None = None
    min_tokens: int | None = None

    stop: str | list[str] | None = None
    stream: bool | None = False

    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    n: int | None = None


class ResponseChoice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: dict | None = None
    finish_reason: FinishReason | None = None


class Response(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str

    choices: list[ResponseChoice]

    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


@api.post("/chat/completions", response_model=Response)
async def create_chat_completion(
    payload: Request,
) -> Response:
    print(payload)

    raise HTTPException(
        status_code=501, detail="Completion logic not implemented"
    )
