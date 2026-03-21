# routers/stt.py
import json
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.runtime import get_whisper_model, run_blocking
from app.core.security import ServiceAuth, sanitize_error_message, sanitize_trace_id
from app.schemas.stt import SttRequest, SttResponse

router = APIRouter()


def _parse_request_payload(raw_payload: str | None, trace_id: str | None) -> SttRequest:
    trace_id = sanitize_trace_id(trace_id)
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
                "reason": sanitize_error_message(str(e), "invalid metadata"),
            },
        ) from e


async def _run_stt(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    request: str | None = Form(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str | None = Form(None),
    _auth: None = ServiceAuth,
):
    req = _parse_request_payload(text or request, traceId)

    if file:
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail={"message": "audio file is required", "traceId": req.traceId},
            )
        audio_path = None
        try:
            raw_bytes = await file.read()
            if len(raw_bytes) > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={"message": "audio file too large", "traceId": req.traceId},
                )
            # Spring에서 wav로 변환해서 보내기로 계약 → suffix wav 고정
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(raw_bytes)
                audio_path = tmp.name

            language = lang or req.lang or "ko"

            try:
                segments, _info = await run_blocking(
                    get_whisper_model().transcribe,
                    audio_path, beam_size=1, language=language
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "STT provider failed",
                        "traceId": req.traceId,
                        "reason": sanitize_error_message(str(e), "stt failed"),
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
    _auth: None = ServiceAuth,
):
    return await _run_stt(file, text, request, mediaUrl, lang, traceId, _auth)


@router.post("/api/worklogs/stt", response_model=SttResponse)
async def stt_legacy(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    request: str | None = Form(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str | None = Form(None),
    _auth: None = ServiceAuth,
):
    return await _run_stt(file, text, request, mediaUrl, lang, traceId, _auth)
