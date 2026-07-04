"""Sailfish data plane — discover and harvest tool-call training data.

Preferred path: the nemesis8 controller (docs/integrations/NEMESIS8_INTEGRATION.md) at SAILFISH_N8_URL. Fallback: the
built-in Python scraper over local Claude Code transcripts (scrape/scrape_toolcalls.py). The UI treats
whatever /sources returns as one list; this module unifies both.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from app.config import settings

_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("SAILFISH_DATA_DIR") or (_ROOT / "data"))
SCRAPE_PY = _ROOT / "scrape" / "scrape_toolcalls.py"


async def probe_nemesis8() -> Dict:
    """Is the nemesis8 controller reachable, and what sources does it have?"""
    url = f"{settings.n8_url.rstrip('/')}/v1/training/sources"
    headers = {}
    if settings.n8_token:
        headers["Authorization"] = f"Bearer {settings.n8_token}"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(url, headers=headers)
        if r.status_code == 200:
            return {"present": True, "url": settings.n8_url, "sources": r.json().get("sources", [])}
        return {"present": False, "url": settings.n8_url, "reason": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"present": False, "url": settings.n8_url, "reason": str(e)[:160]}


def local_stats() -> Optional[Dict]:
    """Stats from the last local scrape (or the committed provenance stats.json)."""
    p = DATA_DIR / "stats.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def local_corpus_present() -> bool:
    p = DATA_DIR / "tool_calls.jsonl"
    return p.exists() and p.stat().st_size > 0


async def list_sources() -> Dict:
    """Unified source list: nemesis8's sources (if up) + the local Claude Code scraper source."""
    n8 = await probe_nemesis8()
    sources: List[Dict] = []
    for s in n8.get("sources", []):
        sources.append({**s, "kind": "nemesis8"})

    st = local_stats()
    local_count = st.get("tool_calls") if st else None
    sources.append({
        "id": "claude-code-local",
        "label": "Claude Code (local transcripts)",
        "tool_calls": local_count,
        "kind": "local-scraper",
        "scraped": local_corpus_present(),
        "note": "built-in fallback; reads ~/.claude transcripts locally, nothing is uploaded",
    })
    return {"nemesis8": {"present": n8["present"], "url": n8.get("url"), "reason": n8.get("reason")},
            "sources": sources}


def run_local_scrape(roots: Optional[List[str]] = None) -> Dict:
    """Invoke the built-in scraper with THIS interpreter (the image is Python-only). Writes into
    DATA_DIR. Returns the scraper's JSON summary. Transcripts must be readable (mounted in the
    appliance; see install scripts)."""
    if not SCRAPE_PY.exists():
        return {"ok": False, "error": f"scraper not found at {SCRAPE_PY}"}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRAPE_PY), "--out", str(DATA_DIR)]
    for r in roots or []:
        cmd += ["--root", r]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "scrape timed out (>600s)"}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-800:]}
    summary = {}
    try:
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    return {"ok": True, "summary": summary, "log": proc.stderr[-800:]}


def export_zip(dest: Optional[Path] = None) -> Path:
    """Zip the harvested corpus for the BYO-cloud training path (data leaves only when the user ships
    it to THEIR cloud). Bundles the jsonl + stats."""
    dest = dest or (DATA_DIR / "sailfish_training_data.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("tool_calls.jsonl", "agentic_prompts.jsonl", "stats.json"):
            p = DATA_DIR / name
            if p.exists():
                z.write(p, arcname=name)
    return dest
