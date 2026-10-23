"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import settings
from app.db.pool import close_pool, init_pool
from app.logging_setup import setup_logging

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(debug=settings.debug)
    init_pool()
    yield
    close_pool()


app = FastAPI(
    title="UITBOT API",
    description="UIT regulation assistant using RAG",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)

if STATIC_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=STATIC_DIR, html=True),
        name="static",
    )
