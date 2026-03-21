import asyncio
from functools import lru_cache

from faster_whisper import WhisperModel
from openai import OpenAI
from transformers import MarianMTModel, MarianTokenizer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel("tiny", device="cpu")


@lru_cache(maxsize=16)
def get_translation_assets(model_name: str) -> tuple[MarianMTModel, MarianTokenizer]:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


async def run_blocking(fn, /, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

