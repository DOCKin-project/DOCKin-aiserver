# schemas/translate.py
from pydantic import BaseModel


class TranslateRequest(BaseModel):
    source: str
    target: str
    traceId: str | None = None
    title: str | None = None
    text: str | None = None
    logText: str | None = None


class TranslateResponse(BaseModel):
    title: str
    translated: str
    model: str
    traceId: str | None = None
