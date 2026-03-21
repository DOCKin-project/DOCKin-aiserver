# schemas/stt.py
from pydantic import BaseModel, Field


class SttRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    logText: str | None = Field(default=None, max_length=5000)
    equipmentId: int | None = None
    imageUrl: str | None = Field(default=None, max_length=500)
    audioFileUrl: str | None = Field(default=None, max_length=255)
    mediaUrl: str | None = Field(default=None, max_length=500)
    lang: str | None = Field(default=None, max_length=10)
    traceId: str | None = Field(default=None, max_length=100)


class SttResponse(BaseModel):
    logId: int | None = None
    userId: str | None = None
    equipmentId: int | None = None
    title: str | None = None
    logText: str = Field(max_length=5000)
    imageUrl: str | None = Field(default=None, max_length=500)
    createdAt: str | None = None
    updatedAt: str | None = None
    audioFileUrl: str | None = Field(default=None, max_length=255)
    provider: str = Field(max_length=50)
    traceId: str | None = Field(default=None, max_length=100)


class RealtimeTranslateResponse(BaseModel):
    originalText: str = Field(max_length=5000)
    translatedText: str = Field(max_length=5000)
    detectedLanguage: str | None = Field(default=None, max_length=20)
    model: str = Field(max_length=100)
    traceId: str | None = Field(default=None, max_length=100)
