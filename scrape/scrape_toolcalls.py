#!/usr/bin/env python3
"""
Sailfish :: scrape (Python twin of scrape_toolcalls.mjs) — harvest tool-call traces from Claude Code
transcripts. This is the copy the APPLIANCE runs (the image is Python-only; no Node), so the data plane
can invoke it with the same interpreter as the gateway. Output is byte-compatible with the .mjs.

Walks ~/.claude/projects (+ the nemesis8 sandbox, + any SAILFISH_TRANSCRIPTS dirs) and extracts every
tool call with its lead-up context, arguments, and result.

  python scrape/scrape_toolcalls.py [--max-context 6] [--out DIR] [--root DIR ...]

Outputs (./data or --out):
  tool_calls.jsonl      one record per tool call: {project, ts, tool, arguments, result_preview, context[]}
  agentic_prompts.jsonl chat-format prompts (context -> expected tool call)
  stats.json            counts by tool, by project, arg-key frequencies
"""
import argparse, json, os, re, sys
from pathlib import Path

# strip ANSI/OSC escapes + control chars (terminal gunk in tool results); keep \t \n
_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_C0 = re.compile(r"[\x1b\r\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ctrl(s):
    if not s:
        return ""
    s = _CSI.sub("", s)
    s = _OSC.sub("", s)
    return _C0.sub("", s)


def clip(s, n):
    s = s or ""
    return s[:n] + "…" if len(s) > n else s


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def walk_jsonl(root):
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jsonl"):
                yield os.path.join(dirpath, name)


def default_roots():
    home = Path.home()
    roots = [
        str(home / ".claude" / "projects"),
        str(home / ".nemesis8" / "home" / ".claude" / "projects"),
    ]
    extra = os.environ.get("SAILFISH_TRANSCRIPTS", "")
    if extra:
        roots = [p for p in extra.split(os.pathsep) if p] + roots
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-context", type=int, default=6)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--root", action="append", default=None, help="transcript root (repeatable)")
    args = ap.parse_args()

    MAX_CTX, CTX_CHARS = args.max_context, 1200
    roots = args.root if args.root else default_roots()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_tool, by_project, arg_keys = {}, {}, {}
    n_files = n_calls = n_lines = n_bad = 0

    files = []
    for r in roots:
        files.extend(walk_jsonl(r))

    with open(out / "tool_calls.jsonl", "w", encoding="utf-8") as tc_out, \
         open(out / "agentic_prompts.jsonl", "w", encoding="utf-8") as pr_out:
        for file in files:
            n_files += 1
            try:
                lines = [ln for ln in Path(file).read_text(encoding="utf-8", errors="replace").split("\n") if ln]
            except Exception:
                continue

            # pass 1: parse events + index tool results by tool_use_id
            events, result_by_id = [], {}
            for line in lines:
                n_lines += 1
                try:
                    o = json.loads(line)
                except Exception:
                    n_bad += 1
                    continue
                m = o.get("message")
                if not m or not m.get("content"):
                    events.append({"role": o.get("type"), "content": "", "project": o.get("cwd"), "ts": o.get("timestamp")})
                    continue
                content = m.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            raw = text_of(b.get("content")) or (b.get("content") if isinstance(b.get("content"), str) else "")
                            result_by_id[b.get("tool_use_id")] = clip(strip_ctrl(raw), 400)
                events.append({"role": m.get("role") or o.get("type"), "content": content,
                               "project": o.get("cwd"), "ts": o.get("timestamp")})

            # pass 2: rolling context -> emit a record at each tool_use
            ctx = []
            for ev in events:
                t = text_of(ev["content"])
                if isinstance(ev["content"], list):
                    for b in ev["content"]:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            project = os.path.basename(ev["project"]) if ev.get("project") else "unknown"
                            inp = b.get("input") or {}
                            rec = {"project": project, "ts": ev.get("ts"), "tool": b.get("name"),
                                   "arguments": inp, "result_preview": result_by_id.get(b.get("id")),
                                   "context": ctx[-MAX_CTX:]}
                            tc_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            pr_out.write(json.dumps({
                                "messages": [{"role": c["role"], "content": c["text"]} for c in ctx[-MAX_CTX:]],
                                "expect_tool": b.get("name"), "arguments": inp, "project": project,
                            }, ensure_ascii=False) + "\n")
                            n_calls += 1
                            name = b.get("name")
                            by_tool[name] = by_tool.get(name, 0) + 1
                            by_project[project] = by_project.get(project, 0) + 1
                            for k in inp.keys():
                                key = f"{name}.{k}"
                                arg_keys[key] = arg_keys.get(key, 0) + 1
                if t and t.strip():
                    role = "assistant" if ev.get("role") == "assistant" else "user"
                    ctx.append({"role": role, "text": clip(strip_ctrl(t).strip(), CTX_CHARS)})
                if len(ctx) > 40:
                    del ctx[: len(ctx) - 40]

    def top(d, n):
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]

    stats = {
        "generated": None,  # stamped by the caller if a clock is available
        "files": n_files, "lines": n_lines, "parse_errors": n_bad, "tool_calls": n_calls,
        "unique_tools": len(by_tool),
        "top_tools": [list(x) for x in top(by_tool, 25)],
        "top_projects": [list(x) for x in top(by_project, 15)],
        "top_arg_keys": [list(x) for x in top(arg_keys, 25)],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Sailfish scrape complete:", file=sys.stderr)
    print(f"  files={n_files} lines={n_lines} parse_errors={n_bad}", file=sys.stderr)
    print(f"  tool_calls={n_calls} unique_tools={stats['unique_tools']}", file=sys.stderr)
    print(json.dumps({"tool_calls": n_calls, "unique_tools": stats["unique_tools"],
                      "files": n_files, "top_tools": stats["top_tools"][:8]}))


if __name__ == "__main__":
    main()
