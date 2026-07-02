"""nuts-auth integration for Sailfish. Mirrors grubcrawler: validate magic-link JWTs
(3-part) via /api/verify, or ahp_ tokens via /auth exchange. Login is a redirect to
auth.nuts.services with return_url back to our callback."""
import logging
from typing import Dict, Optional

import httpx
from fastapi import HTTPException, Header

from app.config import settings

logger = logging.getLogger(__name__)


class AuthClient:
    """Validates nuts-auth JWTs (browser magic-link) or ahp_ API tokens."""

    def __init__(self):
        self.auth_url = settings.gnosis_auth_url.rstrip("/")

    async def validate_token(self, token: str) -> Dict:
        is_jwt = token.count(".") == 2 and not token.startswith("ahp_")
        async with httpx.AsyncClient(timeout=10) as client:
            if is_jwt:
                resp = await client.get(
                    f"{self.auth_url}/api/verify",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                return resp.json()
            resp = await client.post(f"{self.auth_url}/auth", data={"token": token})
            if resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid or inactive token")
            jwt_token = resp.json().get("access_token", "")
            if not jwt_token:
                raise HTTPException(status_code=401, detail="No access token returned")
            verify = await client.get(
                f"{self.auth_url}/api/verify",
                headers={"Authorization": f"Bearer {jwt_token}"},
            )
            if verify.status_code != 200:
                raise HTTPException(status_code=401, detail="Token verification failed")
            return verify.json()


_client = AuthClient()


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Required-auth dependency. 401s if no valid token."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return await _client.validate_token(token)


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    """Optional-auth dependency for pages that render logged-out too."""
    token = _extract_bearer(authorization)
    if not token:
        return None
    try:
        return await _client.validate_token(token)
    except HTTPException:
        return None


def user_email(user: Optional[Dict]) -> Optional[str]:
    if not user:
        return None
    return user.get("email") or user.get("sub") or user.get("hf_user")
