"""Sailfish gateway — one port (22343 local / 8080 Cloud Run).

  /              landing + app UI (static)
  /v1/*          OpenAI-compatible proxy to the serving engine (tools supported)
  /api/status    tier, model, gpu, drafter, live TPS  (for Hyperia's detection ladder)
  /api/gpu       raw GPU probe
  /login         redirect to nuts-auth
  /auth/callback receive the magic-link JWT
  /healthz       liveness
"""
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.auth import get_optional_user, user_email
from app.gpu import detect_gpu, choose_tier, capable_tier

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sailfish")

app = FastAPI(title="sailfish", docs_url=None, redoc_url=None)
SITE = Path(__file__).parent.parent / "site"

_recent_tps: float = 0.0


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "sailfish"}


@app.get("/api/gpu")
async def api_gpu():
    return detect_gpu()


@app.get("/api/status")
async def api_status():
    """The contract Hyperia/nemesis8 read (see HYPERIA_INTEGRATION.md)."""
    gpu = detect_gpu()
    tier, engine, drafter = choose_tier(gpu, settings.tier_override)
    capable = capable_tier(gpu)
    note = None
    if capable == "A" and tier == "B":
        note = ("This card is Tier-A capable (vLLM + MTP drafter, the challenge stack). "
                "The Tier-A image ships in P4; serving Tier B (llama.cpp + n-gram) now.")
    return {
        "service": "sailfish",
        "version": os.environ.get("GIT_SHA", "dev"),
        "tier": tier, "engine": engine, "drafter": drafter,
        "tier_capable": capable,
        "note": note,
        "model": settings.engine_model,
        "gpu": gpu,
        "tps_recent": _recent_tps,
        "hosted": settings.require_auth,
    }


# ---- nuts-auth login (mirrors grubcrawler) ----
@app.get("/login")
async def login():
    ret = f"{settings.public_base_url}/auth/callback"
    return RedirectResponse(url=f"{settings.gnosis_auth_url}/login?return_url={ret}")


@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback():
    # nuts-auth returns the JWT in the URL fragment; a tiny page stashes it and bounces home.
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><title>signing in…</title>"
        "<script>const t=new URLSearchParams(location.hash.slice(1)).get('access_token');"
        "if(t){localStorage.setItem('sailfish_jwt',t);}location.replace('/');</script>"
        "<body style='background:#05070a;color:#7fdfff;font-family:monospace'>signing you in…</body>"
    )


# ---- OpenAI-compatible proxy to the serving engine ----
@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request):
    global _recent_tps
    body = await request.body()
    url = f"{settings.engine_url.rstrip('/')}/{path}"
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.request(
            request.method, url, content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
    dt = time.perf_counter() - t0
    # opportunistic TPS from llama.cpp/vLLM usage, for /api/status
    try:
        j = r.json()
        ct = (j.get("usage") or {}).get("completion_tokens")
        tim = j.get("timings") or {}
        if tim.get("predicted_per_second"):
            _recent_tps = round(float(tim["predicted_per_second"]), 1)
        elif ct and dt > 0:
            _recent_tps = round(ct / dt, 1)
    except Exception:
        pass
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


# ---- static site last, so /api and /v1 win ----
if SITE.exists():
    app.mount("/", StaticFiles(directory=str(SITE), html=True), name="site")
