import logging
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import httpx
from app.config import GEMINI_API_KEY, GEMINI_PROXY, GEMINI_MODEL

router = APIRouter(prefix="/api", tags=["ai"])

logger = logging.getLogger(__name__)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_RETRIES = 2
RETRY_DELAY = 2


class GeminiRequest(BaseModel):
    prompt: str
    api_key: str | None = None


def _extract_error_message(text: str) -> str:
    try:
        import json
        data = json.loads(text)
        return data.get("error", {}).get("message", text[:200])
    except Exception:
        return text[:200]


async def _call_gemini_api(prompt: str, api_key: str | None) -> dict:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API Key 未设置")

    url = f"{GEMINI_API_URL}?key={key}"

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    timeout = httpx.Timeout(60.0, connect=30.0)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if GEMINI_PROXY:
                    response = await client.post(url, json=payload, headers=headers, proxy=GEMINI_PROXY)
                else:
                    response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                return response.json()

            error_detail = _extract_error_message(response.text)
            status_code = response.status_code

            if status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="AI 配额已用尽，请在 Google AI Studio 升级或更换 API Key"
                )

            if status_code == 503:
                if attempt < MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="AI 模型暂时不可用，请稍后重试"
                )

            if status_code == 404:
                raise HTTPException(
                    status_code=400,
                    detail=f"AI 模型不存在或不支持: {GEMINI_MODEL}"
                )

            raise HTTPException(
                status_code=502,
                detail=f"Gemini API 请求失败 ({status_code}): {error_detail}"
            )

        except HTTPException:
            raise
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                import asyncio
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise HTTPException(status_code=504, detail="AI 请求超时，请稍后重试")
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                import asyncio
                await asyncio.sleep(RETRY_DELAY)
                continue

    raise HTTPException(status_code=500, detail=f"AI 服务异常: {last_error}")


@router.post("/ai/gemini")
async def call_gemini(
    req: GeminiRequest,
    x_api_key: str | None = Header(None, alias="x-api-key")
):
    try:
        api_key = req.api_key or x_api_key
        data = await _call_gemini_api(req.prompt, api_key)

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = "**AI 返回格式异常，请稍后重试**"

        return {"result": text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 服务异常: {str(e)[:100]}")
