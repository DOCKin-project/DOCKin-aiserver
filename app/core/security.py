import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.core.config import settings


def sanitize_trace_id(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    cleaned = "".join(ch for ch in trace_id if ch.isalnum() or ch in "-_.")
    return cleaned[:100] if cleaned else None


def sanitize_error_message(message: str, fallback: str) -> str:
    compact = " ".join(message.split()).strip()
    if not compact:
        return fallback
    return compact[:300]


async def verify_service_token(
    x_service_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.service_token
    if not expected:
        return
    if not x_service_token or not secrets.compare_digest(x_service_token, expected):
        raise HTTPException(status_code=401, detail={"message": "Unauthorized"})


ServiceAuth = Depends(verify_service_token)

