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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles

from app import curate as curate_mod
from app import data as data_mod
from app import train as train_mod
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


# ---- appliance guard: in hosted mode, local data/curate ops need a logged-in user ----
async def _require_local(user=Depends(get_optional_user)):
    """Data harvest / curation are appliance features. On the hosted service (require_auth) they must
    not run for anonymous callers — curation spends our curator key, scrape reads a host FS that isn't
    there. Locally (no auth configured) they're open."""
    if settings.require_auth and not user:
        raise HTTPException(status_code=401, detail="login required")
    return user


# ---- data plane (Data view): discover sources, scrape, stats, export ----
@app.get("/api/data/sources")
async def data_sources():
    return await data_mod.list_sources()


@app.get("/api/data/stats")
async def data_stats():
    return data_mod.local_stats() or {"note": "no local corpus yet — run a scrape"}


@app.post("/api/data/scrape")
async def data_scrape(request: Request, _=Depends(_require_local)):
    body = await _json(request)
    return data_mod.run_local_scrape(roots=body.get("roots"))


@app.get("/api/data/export")
async def data_export(_=Depends(_require_local)):
    path = data_mod.export_zip()
    if not path.exists():
        raise HTTPException(status_code=404, detail="nothing to export — scrape or curate first")
    return FileResponse(str(path), media_type="application/zip", filename=path.name)


# ---- curation (Curate view): ESTIMATE is free; RUN spends money and is cap-gated ----
@app.get("/api/curate/estimate")
async def curate_estimate(path: Optional[str] = None, provider: Optional[str] = None,
                          model: Optional[str] = None, cap_usd: Optional[float] = None):
    corpus = path or str(data_mod.DATA_DIR / "tool_calls.jsonl")
    return curate_mod.estimate_cost(corpus, provider=provider, model=model, cap_usd=cap_usd)


@app.post("/api/curate/run")
async def curate_run(request: Request, _=Depends(_require_local)):
    body = await _json(request)
    if not body.get("confirm"):
        raise HTTPException(status_code=400,
                            detail="refused: call /api/curate/estimate and resend with confirm=true + cap_usd")
    corpus = body.get("path") or str(data_mod.DATA_DIR / "tool_calls.jsonl")
    out = body.get("out") or str(data_mod.DATA_DIR / "tool_calls.curated.jsonl")
    return await curate_mod.curate(
        corpus, out, provider=body.get("provider"), model=body.get("model"),
        key=body.get("key"), cap_usd=body.get("cap_usd"),
        batch=int(body.get("batch", 10)), max_examples=body.get("max_examples"),
    )


# ---- training (Train view): BYO Google Cloud — generate script + bundle ----
@app.post("/api/train/byo")
async def train_byo(request: Request, _=Depends(_require_local)):
    body = await _json(request)
    try:
        script = train_mod.generate_byo_script(
            project=body.get("project", ""), region=body.get("region", ""),
            hf_repo=body.get("hf_repo", ""), zone=body.get("zone"),
            base=body.get("base", "google/gemma-4-E4B-it"), epochs=float(body.get("epochs", 1.0)),
            bucket=body.get("bucket"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"script": script, "filename": "train_on_gcloud.sh",
            "bundle_url": "/api/train/bundle",
            "note": "download the bundle, save this script beside it, then run it in your gcloud shell"}


@app.get("/api/train/bundle")
async def train_bundle(_=Depends(_require_local)):
    try:
        path = train_mod.build_training_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(str(path), media_type="application/zip", filename=path.name)


async def _json(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}


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
