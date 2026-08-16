from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .request import Request
from .response import Response

api = APIRouter()


@api.post("/chat/completions", response_model=Response)
async def create_chat_completion(
    payload: Request,
) -> Response:
    print(payload)

    raise HTTPException(
        status_code=501, detail="Completion logic not implemented"
    )
