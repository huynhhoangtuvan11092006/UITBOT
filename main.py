
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router

app = FastAPI(
    title="UITBOT API",
    description="UIT regulation assistant using RAG",
    version="0.1.0",
)

app.include_router(
    chat_router,
    prefix="/api/v1",
    tags=["Chat"],
)

STATIC_DIR = Path(__file__).parent / "app" / "static"

app.mount(
    "/",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="static",
)
