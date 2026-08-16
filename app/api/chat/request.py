from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ToolFunction(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] | None = None


class RequestTool(BaseModel):
    type: Literal["function"]
    function: ToolFunction


class ResponseFormatText(BaseModel):
    type: Literal["text"]


class ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"]


class ResponseFormatJSONSchemaData(BaseModel):
    name: str = Field(min_length=1)
    schema: dict[str, Any]
    strict: bool | None = None


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: ResponseFormatJSONSchemaData


ResponseFormat = (
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema
)


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

    tools: list[RequestTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | None = None
    response_format: ResponseFormat | None = None
