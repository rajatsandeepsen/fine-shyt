from typing import Literal

from pydantic import BaseModel

FinishReason = Literal[
    "stop",
    "length",
    "content_filter",
    "function_call",
    "tool_calls",
    "error",
]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None


class ResponseChoice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: dict | None = None
    finish_reason: FinishReason | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Response(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str

    choices: list[ResponseChoice]

    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
