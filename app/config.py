"""Sailfish config. Env-driven; mirrors the nuts.services family conventions."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # nuts-auth (login + token verification)
    gnosis_auth_url: str = os.environ.get("GNOSIS_AUTH_URL", "https://auth.nuts.services")
    public_base_url: str = os.environ.get("PUBLIC_BASE_URL", "https://sailfish.nuts.services")

    # local appliance gateway
    sailfish_port: int = int(os.environ.get("SAILFISH_PORT", "22343"))
    engine_url: str = os.environ.get("SAILFISH_ENGINE_URL", "http://localhost:8080/v1")
    engine_model: str = os.environ.get("SAILFISH_MODEL", "gemma4-e4b")

    # tier / gpu
    tier_override: str = os.environ.get("SAILFISH_TIER", "auto")  # auto|A|B

    # nemesis8 data plane (host-exposed controller)
    n8_url: str = os.environ.get("SAILFISH_N8_URL", "http://host.docker.internal:18042")
    n8_token: str = os.environ.get("SAILFISH_N8_TOKEN", "")  # optional bearer (local, so usually empty)

    # curation frontier model (user brings a token)
    curator_provider: str = os.environ.get("SAILFISH_CURATOR", "anthropic")
    curator_key: str = os.environ.get("SAILFISH_CURATOR_KEY", "")
    curator_cost_cap_usd: float = float(os.environ.get("SAILFISH_COST_CAP_USD", "5"))

    # hosted-mode auth gate; empty = dev/local open mode
    auth_jwks_url: str = os.environ.get("NUTS_AUTH_JWKS_URL", "")

    @property
    def require_auth(self) -> bool:
        return bool(self.auth_jwks_url)


settings = Settings()
