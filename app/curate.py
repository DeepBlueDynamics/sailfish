"""Sailfish curation — a frontier model filters raw tool-call traces into clean training data.

DESIGN (cost is the constraint — PLAN.md §6.2, §11):
  * ESTIMATE FIRST, ALWAYS. estimate_cost() makes NO API calls; the UI shows $ before any token leaves.
  * The cap is enforced DURING the run: after each batch we add the real usage and STOP before the next
    batch would exceed cap_usd. Default cap $5 (settings.curator_cost_cap_usd), user-raisable.
  * Curation only FILTERS (keep/drop + reason). It emits the SAME {context,tool,arguments} schema as the
    scraper — serving-format rendering stays in train/finetune_target.py (render()), not duplicated here.
  * Providers: Anthropic (default) + any OpenAI-compatible endpoint (OpenAI, GLM, Gemini's OpenAI shim).

This module is import-safe with no key configured; nothing runs until curate() is called explicitly.
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import httpx

from app.config import settings

# Rough public list prices, USD per 1M tokens (input, output). ESTIMATES — refresh as prices move.
PRICING = {
    "anthropic:claude-haiku-4-5": (1.0, 5.0),
    "anthropic:claude-sonnet-5": (3.0, 15.0),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4o": (2.50, 10.0),
    "glm:glm-4-flash": (0.10, 0.10),
    "gemini:gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_MODEL = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini",
                  "glm": "glm-4-flash", "gemini": "gemini-2.0-flash"}
# OpenAI-compatible base URLs for the non-anthropic providers
_OPENAI_BASE = {
    "openai": "https://api.openai.com/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

_SYSTEM = (
    "You audit tool-call traces for training a small tool-calling model. For each example decide if it is "
    "a CLEAN training signal: the user intent in the context is clear, and the chosen tool + arguments are "
    "a correct, unambiguous response. DROP examples that are truncated, error-recovery retries, ambiguous, "
    "or where the context doesn't justify the call. Reply ONLY with a JSON array, one object per example in "
    "order: {\"i\": <index>, \"keep\": <bool>, \"reason\": \"<short>\"}."
)


def _price(provider: str, model: str) -> Tuple[float, float]:
    return PRICING.get(f"{provider}:{model}", (1.0, 5.0))  # conservative default if unknown


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)  # ~4 chars/token, deliberately rough


def _example_text(rec: Dict) -> str:
    ctx = " ".join(c.get("text", "") for c in (rec.get("context") or []))
    return f"{ctx} {rec.get('tool','')} {json.dumps(rec.get('arguments') or {})}"


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tool"):
            rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def estimate_cost(corpus_path: str, provider: Optional[str] = None, model: Optional[str] = None,
                  batch: int = 10, cap_usd: Optional[float] = None) -> Dict:
    """NO API CALLS. Estimate the spend to curate the whole corpus at current prices."""
    provider = provider or settings.curator_provider
    model = model or _DEFAULT_MODEL.get(provider, "unknown")
    cap_usd = settings.curator_cost_cap_usd if cap_usd is None else cap_usd
    rows = load_jsonl(Path(corpus_path))
    in_tok = sum(_approx_tokens(_example_text(r)) for r in rows)
    in_tok += _approx_tokens(_SYSTEM) * ((len(rows) // max(1, batch)) + 1)  # system repeated per batch
    out_tok = len(rows) * 24  # ~24 tokens/verdict
    pin, pout = _price(provider, model)
    est = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return {
        "examples": len(rows), "provider": provider, "model": model,
        "est_input_tokens": in_tok, "est_output_tokens": out_tok,
        "est_usd": round(est, 4), "cap_usd": cap_usd, "within_cap": est <= cap_usd,
        "note": "estimate only — no API calls made; prices are approximate list prices",
    }


async def _call(provider: str, key: str, model: str, user: str) -> Tuple[str, int, int]:
    """One frontier call. Returns (text, input_tokens, output_tokens)."""
    if provider == "anthropic":
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": model, "max_tokens": 2048, "system": _SYSTEM,
                      "messages": [{"role": "user", "content": user}]},
            )
        r.raise_for_status()
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage", {})
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)

    base = _OPENAI_BASE.get(provider)
    if not base:
        raise ValueError(f"unknown curator provider: {provider}")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "max_tokens": 2048,
                  "messages": [{"role": "system", "content": _SYSTEM},
                               {"role": "user", "content": user}]},
        )
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    u = j.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _batch_prompt(batch: List[Dict]) -> str:
    lines = []
    for i, rec in enumerate(batch):
        ctx = " | ".join(f"{c.get('role')}: {c.get('text','')[:400]}" for c in (rec.get("context") or [])[-4:])
        lines.append(f"[{i}] context: {ctx}\n    tool: {rec.get('tool')}  args: {json.dumps(rec.get('arguments') or {})[:400]}")
    return "Examples:\n" + "\n".join(lines)


async def curate(corpus_path: str, out_path: str, provider: Optional[str] = None,
                 model: Optional[str] = None, key: Optional[str] = None,
                 cap_usd: Optional[float] = None, batch: int = 10,
                 max_examples: Optional[int] = None) -> Dict:
    """Filter the corpus with the frontier model, enforcing the $ cap. Writes kept examples (same
    {context,tool,arguments} schema) to out_path. Returns a review summary for the UI."""
    provider = provider or settings.curator_provider
    model = model or _DEFAULT_MODEL.get(provider, "")
    key = key or settings.curator_key
    cap_usd = settings.curator_cost_cap_usd if cap_usd is None else cap_usd
    if not key:
        return {"ok": False, "error": "no curator key configured (SAILFISH_CURATOR_KEY)"}

    rows = load_jsonl(Path(corpus_path), limit=max_examples)
    pin, pout = _price(provider, model)
    spent = 0.0
    seen = kept = 0
    stopped = "completed"
    samples: List[Dict] = []  # a few kept/dropped decisions for the review UI

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for start in range(0, len(rows), batch):
            # halt BEFORE spending if we're already at/over cap
            if spent >= cap_usd:
                stopped = "cap_reached"
                break
            chunk = rows[start:start + batch]
            try:
                text, itok, otok = await _call(provider, key, model, _batch_prompt(chunk))
            except Exception as e:
                stopped = f"error: {str(e)[:160]}"
                break
            spent += itok / 1e6 * pin + otok / 1e6 * pout
            verdicts = _parse_verdicts(text, len(chunk))
            for i, rec in enumerate(chunk):
                seen += 1
                v = verdicts.get(i, {"keep": True, "reason": "unscored (kept by default)"})
                if v.get("keep"):
                    out.write(json.dumps({"context": rec.get("context"), "tool": rec.get("tool"),
                                          "arguments": rec.get("arguments")}, ensure_ascii=False) + "\n")
                    kept += 1
                if len(samples) < 40:
                    samples.append({"tool": rec.get("tool"), "keep": bool(v.get("keep")),
                                    "reason": v.get("reason", "")})

    return {"ok": True, "provider": provider, "model": model, "seen": seen, "kept": kept,
            "dropped": seen - kept, "spent_usd": round(spent, 4), "cap_usd": cap_usd,
            "stopped": stopped, "out": out_path, "samples": samples}


def _parse_verdicts(text: str, n: int) -> Dict[int, Dict]:
    """Best-effort parse of the model's JSON array; missing entries default to keep."""
    out: Dict[int, Dict] = {}
    try:
        s = text[text.index("["): text.rindex("]") + 1]
        for obj in json.loads(s):
            if isinstance(obj, dict) and "i" in obj:
                out[int(obj["i"])] = {"keep": bool(obj.get("keep", True)), "reason": str(obj.get("reason", ""))[:120]}
    except Exception:
        pass
    return out
