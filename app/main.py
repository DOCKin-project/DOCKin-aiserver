# main.py
from fastapi import FastAPI
from app.routers import health, chatbot, sync_translate, stt

app = FastAPI(
    title="fastapi-ai",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(health.router)
app.include_router(chatbot.router)
app.include_router(sync_translate.router)
app.include_router(stt.router)


@app.get("/")
def root():
    return {"hello": "fastapi-ai"}
