# schemas/translate.py
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    source: str = Field(min_length=2, max_length=10)
    target: str = Field(min_length=2, max_length=10)
    traceId: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=5000)
    logText: str | None = Field(default=None, max_length=5000)


class TranslateResponse(BaseModel):
    title: str = Field(max_length=500)
    translated: str = Field(max_length=5000)
    model: str = Field(max_length=100)
    traceId: str | None = Field(default=None, max_length=100)
