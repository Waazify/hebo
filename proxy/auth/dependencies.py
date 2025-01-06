from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from .utils import verify_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key_organization(api_key: str | None = Security(api_key_header)):
    """
    Dependency to validate API key and return organization
    """
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key"
        )

    organization = await verify_api_key(api_key)
    if not organization:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key"
        )

    return organization
