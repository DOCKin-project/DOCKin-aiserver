# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import settings
from app.routers import health, chatbot, sync_translate, stt


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_upload_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"message": "Request entity too large"},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Invalid Content-Length"},
                )
        return await call_next(request)

app = FastAPI(
    title="fastapi-ai",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or [],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Service-Token"],
)

app.include_router(health.router)
app.include_router(chatbot.router)
app.include_router(sync_translate.router)
app.include_router(stt.router)


@app.get("/")
def root():
    return {"hello": "fastapi-ai"}
