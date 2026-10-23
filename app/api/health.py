"""Liveness and database connectivity."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.db.pool import ping_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    try:
        ping_db()
    except Exception:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Database unavailable.") from None
    return {"status": "ok", "database": "connected"}
