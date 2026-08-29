from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

from .api.chat.main import api as chat

api = FastAPI(title="Inference API", version="1.0.0")
api.include_router(chat)


@api.get("/")
def root() -> dict[str, Any]:
    paths = [
        route.path
        for router in (chat, api)
        for route in router.routes
        if isinstance(route, APIRoute)
    ]
    return {
        "message": "Inference server is running",
        "endpoints": paths,
    }
