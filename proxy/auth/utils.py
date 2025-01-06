from typing import Optional
import httpx
from config import settings


async def verify_api_key(api_key: str) -> Optional[dict]:
    """
    Verify API key against Django backend
    Returns organization dict if valid, None if invalid
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{settings.BACKEND_URL}/api/verify-key/",
                headers={"X-API-Key": api_key},
            )
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.RequestError:
            return None
