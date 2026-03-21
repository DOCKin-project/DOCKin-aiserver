# routers/sync_translate.py
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.runtime import (
    get_translation_assets,
    get_whisper_model,
    run_blocking,
)
from app.core.security import ServiceAuth, sanitize_error_message, sanitize_trace_id
from app.schemas.stt import RealtimeTranslateResponse
from app.schemas.translate import TranslateRequest, TranslateResponse

router = APIRouter()

# 지원 언어쌍별 MarianMT 모델 매핑
MODEL_NAME_MAP: dict[tuple[str, str], str] = {
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "th"): "Helsinki-NLP/opus-mt-en-th",
    ("th", "en"): "Helsinki-NLP/opus-mt-th-en",
}

# 로드된 모델과 토크나이저 캐시
def get_model_name(src: str, tgt: str) -> str:
    key = (src, tgt)
    if key not in MODEL_NAME_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 번역 언어쌍입니다. source={src}, target={tgt}",
        )
    return MODEL_NAME_MAP[key]


def _translate_sync(text: str, model_name: str) -> str:
    model, tokenizer = get_translation_assets(model_name)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    generated = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True,
    )

    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translated


async def translate_with_model(text: str, source: str, target: str) -> tuple[str, str]:
    model_name = get_model_name(source, target)
    try:
        translated = await run_blocking(_translate_sync, text, model_name)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Translate model load failed",
                "model": model_name,
                "reason": sanitize_error_message(str(e), "translate failed"),
            },
        ) from e
    return translated, model_name


async def _translate(req: TranslateRequest):
    req.traceId = sanitize_trace_id(req.traceId)
    source_text = req.logText or req.text
    if not source_text:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "text or logText is required",
                "traceId": req.traceId,
            },
        )

    if not req.title:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "title is required",
                "traceId": req.traceId,
            },
        )

    title_translated, title_model_name = await translate_with_model(
        req.title, req.source, req.target
    )
    body_translated, body_model_name = await translate_with_model(
        source_text, req.source, req.target
    )

    model_name = body_model_name or title_model_name

    return TranslateResponse(
        title=title_translated,
        translated=body_translated,
        model=model_name,
        traceId=req.traceId,
    )


@router.post("/api/ai/translate", response_model=TranslateResponse)
async def translate_ai(
    req: TranslateRequest,
    _auth: None = ServiceAuth,
):
    try:
        return await _translate(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Translate failed",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "translate failed"),
            },
        )


@router.post("/api/translate", response_model=TranslateResponse)
async def translate_legacy(
    req: TranslateRequest,
    _auth: None = ServiceAuth,
):
    try:
        return await _translate(req)

    except HTTPException:
        # 지원 언어쌍 오류(400) 등은 그대로 전달
        raise
    except Exception as e:
        # 추론/토크나이징 등 기타 오류
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Translate failed",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "translate failed"),
            },
        )


@router.post("/api/ai/rt-translate", response_model=RealtimeTranslateResponse)
async def realtime_translate(
    file: UploadFile = File(...),
    source: str = Form(...),
    target: str = Form(...),
    traceId: str | None = Form(None),
    _auth: None = ServiceAuth,
):
    traceId = sanitize_trace_id(traceId)
    audio_path = None

    try:
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail={"message": "audio file is required", "traceId": traceId},
            )
        raw_bytes = await file.read()
        if len(raw_bytes) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail={"message": "audio file too large", "traceId": traceId},
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_bytes)
            audio_path = tmp.name

        segments, info = await run_blocking(
            get_whisper_model().transcribe, audio_path, beam_size=1, language=source
        )
        original_text = "".join([seg.text for seg in segments]).strip()

        if not original_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No speech detected",
                    "traceId": traceId,
                },
            )

        translated_text, model_name = await translate_with_model(original_text, source, target)

        return RealtimeTranslateResponse(
            originalText=original_text,
            translatedText=translated_text,
            detectedLanguage=getattr(info, "language", None),
            model=model_name,
            traceId=traceId,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Realtime translate failed",
                "traceId": traceId,
                "reason": sanitize_error_message(str(e), "realtime translate failed"),
            },
        )
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
