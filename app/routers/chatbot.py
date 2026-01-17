# routers/chatbot.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings

router = APIRouter()
client = OpenAI(api_key=settings.openai_api_key)

bearer_scheme = HTTPBearer(auto_error=False)


@router.post("/api/chatbot", response_model=ChatResponse)
async def chatbot(
    req: ChatRequest,
    _cred: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail={"message": "OPENAI_API_KEY not configured", "traceId": req.traceId},
        )

    try:
        completion = client.chat.completions.create(
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
                "reason": str(e),
            },
        )
    except (APITimeoutError, APIConnectionError) as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI connection failed",
                "traceId": req.traceId,
                "reason": str(e),
            },
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI API error",
                "traceId": req.traceId,
                "reason": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Chatbot failed",
                "traceId": req.traceId,
                "reason": str(e),
            },
        )
