from pathlib import Path
from typing import cast

from jinja2 import Template
from pydantic import BaseModel

from app.api.chat.request import Message, RequestTool, ToolFunction

parent_path = Path(__file__).resolve().parent
template_path = parent_path / "template.jinja"
chat_template = template_path.read_text(encoding="utf-8")
template = cast(Template, Template(chat_template))


class Data(BaseModel):
    messages: list[Message]
    tools: list[RequestTool]


data = Data(
    messages=[
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hey. What do you want?"),
    ],
    # tools=[],
    tools=[
        RequestTool(
            type="function",
            function=ToolFunction(
                name="getWeather", description="", parameters={}
            ),
        )
    ],
)

raw_data: dict = data.model_dump(mode="json")
# print(raw_data)

print(template.render(**raw_data).strip())
