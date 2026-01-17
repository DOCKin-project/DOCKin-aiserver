# routers/stt.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.schemas.stt import SttResponse
from faster_whisper import WhisperModel
import tempfile
import os

router = APIRouter()

model = WhisperModel("tiny", device="cpu")
bearer_scheme = HTTPBearer(auto_error=False)


@router.post("/api/worklogs/stt", response_model=SttResponse)
async def stt(
    file: UploadFile | None = File(None),
    mediaUrl: str | None = Form(None),
    lang: str | None = Form(None),
    traceId: str = Form(...),
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),  # 전달만 받음
):
    if file:
        audio_path = None
        try:
            # Spring에서 wav로 변환해서 보내기로 계약 → suffix wav 고정 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                audio_path = tmp.name

            language = lang or "ko"

            try:
                segments, _info = model.transcribe(
                    audio_path, beam_size=1, language=language
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "STT provider failed",
                        "traceId": traceId,
                        "reason": str(e),
                    },
                )

            text = "".join([seg.text for seg in segments]).strip()
            return SttResponse(text=text, provider="whisper-faster", traceId=traceId)

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    if mediaUrl:
        raise HTTPException(
            status_code=400,
            detail={"message": "mediaUrl STT is not supported", "traceId": traceId},
        )

    raise HTTPException(
        status_code=400,
        detail={"message": "file is required", "traceId": traceId},
    )
