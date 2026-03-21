# schemas/chat.py
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=30)
    domain: str | None = Field(default=None, max_length=50)
    lang: str | None = Field(default=None, max_length=10)
    traceId: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    reply: str
    model: str
    traceId: str | None = None
