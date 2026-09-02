from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter

from src.fine_tune.config import Config
from src.fine_tune.inference import Inference

from .request import Request
from .response import Response, ResponseChoice

api = APIRouter()


@api.post("/chat/completions", response_model=Response)
async def create_chat_completion(
    payload: Request,
):
    messages = Inference(payload.messages)

    return Response(
        id=uuid4().hex,
        model=Config.model_name,
        object="chat.completion",
        created=int(time.time()),
        choices=[ResponseChoice(index=1, message=m) for m in messages],
    )
