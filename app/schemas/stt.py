# schemas/stt.py
from pydantic import BaseModel


class SttRequest(BaseModel):
    title: str | None = None
    logText: str | None = None
    equipmentId: int | None = None
    imageUrl: str | None = None
    audioFileUrl: str | None = None
    mediaUrl: str | None = None
    lang: str | None = None
    traceId: str | None = None


class SttResponse(BaseModel):
    logId: int | None = None
    userId: str | None = None
    equipmentId: int | None = None
    title: str | None = None
    logText: str
    imageUrl: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None
    audioFileUrl: str | None = None
    provider: str
    traceId: str | None = None


class RealtimeTranslateResponse(BaseModel):
    originalText: str
    translatedText: str
    detectedLanguage: str | None = None
    model: str
    traceId: str | None = None
