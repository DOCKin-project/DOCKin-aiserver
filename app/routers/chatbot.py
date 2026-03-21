# routers/chatbot.py
from fastapi import APIRouter, Depends, HTTPException
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings
from app.core.runtime import get_openai_client, run_blocking
from app.core.security import ServiceAuth, sanitize_error_message, sanitize_trace_id

router = APIRouter()


@router.post("/api/chatbot", response_model=ChatResponse)
async def chatbot(
    req: ChatRequest,
    _auth: None = ServiceAuth,
):
    req.traceId = sanitize_trace_id(req.traceId)
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail={"message": "OPENAI_API_KEY not configured", "traceId": req.traceId},
        )

    try:
        completion = await run_blocking(
            get_openai_client().chat.completions.create,
            model=settings.openai_model,
            messages=[m.model_dump() for m in req.messages],
        )
        reply = completion.choices[0].message.content or ""
        return ChatResponse(
            reply=reply, model=settings.openai_model, traceId=req.traceId
        )

    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "OpenAI rate limited",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "rate limited"),
            },
        )
    except (APITimeoutError, APIConnectionError) as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI connection failed",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "connection failed"),
            },
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI API error",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "api error"),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Chatbot failed",
                "traceId": req.traceId,
                "reason": sanitize_error_message(str(e), "chatbot failed"),
            },
        )
