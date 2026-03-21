# routers/sync_translate.py
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

from app.schemas.stt import RealtimeTranslateResponse
from app.schemas.translate import TranslateRequest, TranslateResponse

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)
stt_model = WhisperModel("tiny", device="cpu")

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
model_cache: dict[str, MarianMTModel] = {}
tokenizer_cache: dict[str, MarianTokenizer] = {}


def get_model_and_tokenizer(
    src: str, tgt: str
) -> tuple[MarianMTModel, MarianTokenizer, str]:
    key = (src, tgt)
    if key not in MODEL_NAME_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 번역 언어쌍입니다. source={src}, target={tgt}",
        )

    model_name = MODEL_NAME_MAP[key]

    # 캐시에 없으면 로드
    if model_name not in model_cache:
        try:
            tokenizer_cache[model_name] = MarianTokenizer.from_pretrained(model_name)
            model_cache[model_name] = MarianMTModel.from_pretrained(model_name)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Translate model load failed",
                    "model": model_name,
                    "reason": str(e),
                },
            )

    return model_cache[model_name], tokenizer_cache[model_name], model_name


def translate_with_model(text: str, source: str, target: str) -> tuple[str, str]:
    model, tokenizer, model_name = get_model_and_tokenizer(source, target)
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
    return translated, model_name


async def _translate(req: TranslateRequest):
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

    title_translated, title_model_name = translate_with_model(
        req.title, req.source, req.target
    )
    body_translated, body_model_name = translate_with_model(
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
    _cred: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),  # JWT 전달만 받음
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
                "reason": str(e),
            },
        )


@router.post("/api/translate", response_model=TranslateResponse)
async def translate_legacy(
    req: TranslateRequest,
    _cred: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),  # JWT 전달만 받음
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
                "reason": str(e),
            },
        )


@router.post("/api/ai/rt-translate", response_model=RealtimeTranslateResponse)
async def realtime_translate(
    file: UploadFile = File(...),
    source: str = Form(...),
    target: str = Form(...),
    traceId: str | None = Form(None),
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        segments, info = stt_model.transcribe(audio_path, beam_size=1, language=source)
        original_text = "".join([seg.text for seg in segments]).strip()

        if not original_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No speech detected",
                    "traceId": traceId,
                },
            )

        translated_text, model_name = translate_with_model(original_text, source, target)

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
                "reason": str(e),
            },
        )
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
