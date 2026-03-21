# routers/stt.py
import json
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel

from app.schemas.stt import SttRequest, SttResponse

router = APIRouter()

model = WhisperModel("tiny", device="cpu")
bearer_scheme = HTTPBearer(auto_error=False)


def _parse_request_payload(raw_payload: str | None, trace_id: str | None) -> SttRequest:
    if not raw_payload:
        return SttRequest(traceId=trace_id)
    try:
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict):
            raise ValueError("request payload must be an object")
        return SttRequest(**payload, traceId=trace_id or payload.get("traceId"))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid STT request metadata",
                "traceId": trace_id,
                "reason": str(e),
            },
        ) from e


async def _run_stt(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    request: str | None = Form(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str | None = Form(None),
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),  # 전달만 받음
):
    req = _parse_request_payload(text or request, traceId)

    if file:
        audio_path = None
        try:
            # Spring에서 wav로 변환해서 보내기로 계약 → suffix wav 고정
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                audio_path = tmp.name

            language = lang or req.lang or "ko"

            try:
                segments, _info = model.transcribe(
                    audio_path, beam_size=1, language=language
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "STT provider failed",
                        "traceId": req.traceId,
                        "reason": str(e),
                    },
                )

            text = "".join([seg.text for seg in segments]).strip()
            return SttResponse(
                equipmentId=req.equipmentId,
                title=req.title,
                logText=text,
                imageUrl=req.imageUrl,
                audioFileUrl=req.audioFileUrl or file.filename,
                provider="whisper-faster",
                traceId=req.traceId,
            )

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    if mediaUrl:
        raise HTTPException(
            status_code=400,
            detail={"message": "mediaUrl STT is not supported", "traceId": req.traceId},
        )

    raise HTTPException(
        status_code=400,
        detail={"message": "file is required", "traceId": req.traceId},
    )


@router.post("/api/work-logs/stt", response_model=SttResponse)
async def stt_hyphenated(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    request: str | None = Form(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str | None = Form(None),
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    return await _run_stt(file, text, request, mediaUrl, lang, traceId, _cred)


@router.post("/api/worklogs/stt", response_model=SttResponse)
async def stt_legacy(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    request: str | None = Form(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str | None = Form(None),
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    return await _run_stt(file, text, request, mediaUrl, lang, traceId, _cred)
