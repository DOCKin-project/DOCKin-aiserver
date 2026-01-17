# routers/sync_translate.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.schemas.translate import TranslateRequest, TranslateResponse
from transformers import MarianMTModel, MarianTokenizer

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)

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


@router.post("/api/translate", response_model=TranslateResponse)
async def translate(
    req: TranslateRequest,
    _cred: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),  # JWT 전달만 받음
):
    try:
        model, tokenizer, model_name = get_model_and_tokenizer(req.source, req.target)

        inputs = tokenizer(
            req.text,
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

        return TranslateResponse(
            translated=translated, model=model_name, traceId=req.traceId
        )

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
