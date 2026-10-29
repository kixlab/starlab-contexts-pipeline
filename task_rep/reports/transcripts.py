"""
Full transcripts of finished runs, for reports (no model calls).

Pipeline runs: every proposal request and its raw response, read from the
request logs in ``static/lm-batches`` (``utils.llm.BATCH_PATH``) that each
step recorded (``runner_state["batches"]``).

Agent runs: the stored conversation, untruncated, with the system prompt and
tool specs the run recorded.

Long texts are outlined into titled sections (prompt headings) and children
(one per video, factor, or tried factor) so the page can disclose them
progressively.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from facets.prompts.fa_facet_iterative import (
    SECTION_FACETS, SECTION_HISTORY, SECTION_ITEMS, SECTION_SCHEMAS)
from utils.llm import llm_get_requests_path, llm_get_responses_path

from .runs import LoadedRun

# Prompt headings: a section template's first line ("- Contextual factors in
# the pool:"), any "- ...:" line, a capitalized heading ("HOW TO WORK"), and
# the design-space summary ("Design space: kept 6 of 7 ...").
_SECTIONS = (SECTION_FACETS, SECTION_ITEMS, SECTION_SCHEMAS, SECTION_HISTORY)
_KNOWN = {t.split("\n", 1)[0] for t in _SECTIONS}
_HEADING = re.compile(r"^(?:- [^\n]{2,200}:|[A-Z][A-Z &/-]{2,}(?: \([^\n]*\))?"
                      r"|Design space: [^\n]*)$")
_VIDEO = re.compile(r"^Video (\S+):$", re.M)
_SUBHEADING = re.compile(r"^[A-Z][a-z][^\n:]{0,60}:$", re.M)   # "Kept combinations:"
_FACTOR = re.compile(r"<factor id=\"([^\"]+)\">.*?</factor>", re.S)
_TRIED = re.compile(r"<tried_factor>.*?</tried_factor>", re.S)


def _tag(text: str, tag: str) -> str:
    m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""


def _children(body: str) -> Optional[Dict[str, Any]]:
    """Split a section body into one child per video, factor, or tried factor."""
    starts = [m.start() for m in _VIDEO.finditer(body)]
    if starts:
        parts = [body[a:b].strip("\n") for a, b in zip(starts, starts[1:] + [len(body)])]
        children = []
        for part in parts:
            head, _, rest = part.partition("\n")
            n = len(re.findall(r"<segment |^S\d+:", rest, re.M))
            children.append({"title": head.rstrip(":") + (f" · {n} segments" if n else ""),
                             "text": rest})
        return {"intro": body[:starts[0]].strip("\n"), "children": children}
    starts = [m.start() for m in _SUBHEADING.finditer(body)]
    if starts:
        children = []
        for a, b in zip(starts, starts[1:] + [len(body)]):
            head, _, rest = body[a:b].strip("\n").partition("\n")
            n = len(re.findall(r"^- (?!none$)", rest, re.M))
            children.append({"title": head.rstrip(":") + (f" ({n})" if n else ""),
                             "text": rest})
        return {"intro": body[:starts[0]].strip("\n"), "children": children}
    for pattern, title in ((_FACTOR, lambda m: f"{m.group(1)} · {_tag(m.group(0), 'title')}"),
                           (_TRIED, lambda m: f"{_tag(m.group(0), 'title')} — "
                                              f"{_tag(m.group(0), 'why_not_kept')}")):
        matches = list(pattern.finditer(body))
        if matches:
            rest = pattern.sub("", body).strip("\n")
            return {"intro": rest, "children": [{"title": title(m), "text": m.group(0)}
                                                for m in matches]}
    return None


def outline(text: str) -> List[Dict[str, Any]]:
    """``text`` as sections ``{"title", "text", "children"?}``; the first
    section's title is None when the text starts without a heading."""
    sections, title, lines = [], None, []

    def close():
        body = "\n".join(lines).strip("\n")
        if title is None and not body:
            return
        section: Dict[str, Any] = {"title": title, "text": body}
        split = _children(body)
        if split:
            section.update(text=split["intro"], children=split["children"])
        sections.append(section)

    for line in text.split("\n"):
        if line in _KNOWN and not line.endswith(":"):
            # a heading sentence: its first clause is the title, the rest is text
            close()
            head, _, rest = line.lstrip("- ").partition(". ")
            title, lines = head, [rest] if rest else []
        elif _HEADING.match(line):
            close()
            title, lines = line.rstrip(":").lstrip("- "), []
        else:
            lines.append(line)
    close()
    return sections


def _message_text(content: Any) -> str:
    """A logged message's text (``MessageContent.to_dict()`` or a string)."""
    if isinstance(content, str):
        return content
    blocks = content.get("content", []) if isinstance(content, dict) else content
    return "".join(b.get("text", "") if b.get("type") == "text" else f"[{b.get('type')}]"
                   for b in blocks)


def _read_jsonl(path: str) -> List[Any]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _request(batch_id: str) -> Dict[str, Any]:
    """One logged proposal request: its messages, response, and usage."""
    req_path, res_path = llm_get_requests_path(batch_id), llm_get_responses_path(batch_id)
    if not Path(req_path).exists():
        return {"batch": batch_id, "missing": True}
    request = _read_jsonl(req_path)[0]
    responses = _read_jsonl(res_path) if Path(res_path).exists() else []
    record = next((r for r in responses if r[0].get("request_id") == request["request_id"]),
                  responses[0] if responses else None)
    messages = {m["role"]: _message_text(m["content"]) for m in request["messages"]}
    return {
        "batch": batch_id,
        "seed": request.get("options", {}).get("seed"),
        "system": messages.get("system", ""),
        "user": messages.get("user", ""),
        "response": record[1] if record else None,
        "usage": record[2] if record and len(record) > 2 else None,
    }


def pipeline_transcript(run: LoadedRun) -> Optional[Dict[str, Any]]:
    """Every proposal request and response of a pipeline run, or None when the
    run recorded no request logs."""
    batches = [batch for step in run.result.runner_state.get("batches", [])
               for batch in step.get("proposal", [])]
    if not batches:
        return None
    requests = [_request(b) for b in batches]
    found = [r for r in requests if not r.get("missing")]
    systems = {r["system"] for r in found}
    shared = systems.pop() if len(systems) == 1 else None

    unused = list(run.result.ledger)   # outcome of each proposed factor, by title
    steps = []
    for n, r in enumerate(requests, 1):
        if r.get("missing"):
            steps.append({"n": n, "batch": r["batch"], "missing": True})
            continue
        response = r["response"] if isinstance(r["response"], dict) else {}
        candidates = []
        for c in response.get("candidates", []):
            match = next((x for x in unused if x.title == c.get("title")), None)
            if match is not None:
                unused.remove(match)
            candidates.append({"factor": c, "status": match.status.value if match else None})
        steps.append({
            "n": n, "batch": r["batch"], "seed": r["seed"], "usage": r["usage"],
            "system": None if shared is not None else outline(r["system"]),
            "user": outline(r["user"]), "user_chars": len(r["user"]),
            "candidates": candidates,
            "other": {k: v for k, v in response.items() if k != "candidates"},
            "raw": json.dumps(r["response"], ensure_ascii=False, indent=2),
        })
    return {
        "kind": "pipeline",
        "system": outline(shared) if shared is not None else None,
        "system_chars": len(shared) if shared is not None else None,
        "steps": steps,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def agentic_transcript(run: LoadedRun) -> Optional[Dict[str, Any]]:
    """The agent's whole conversation with its system prompt and tools."""
    rs = run.result.runner_state
    if not rs.get("transcript"):
        return None
    system, tools = rs.get("system"), rs.get("tools")
    note = "" if system else "This run did not record its system prompt."
    messages = []
    for m in rs["transcript"]:
        if m["role"] == "user":
            messages.append({"role": "user", "text": m["text"]})
        elif m["role"] == "assistant":
            messages.append({"role": "agent", "text": m.get("text", ""),
                             "calls": [{"name": c["name"], "args": c["arguments"]}
                                       for c in m.get("tool_calls", [])]})
        else:
            messages.append({"role": "tool", "name": m["name"], "text": m["content"],
                             "sections": outline(m["content"])})
    return {
        "kind": "agentic",
        "system": outline(system) if system else None,
        "system_chars": len(system) if system else None,
        "tools": tools or [],
        "messages": messages,
        "note": note,
    }


def transcript(run: LoadedRun) -> Optional[Dict[str, Any]]:
    if run.kind == "pipeline":
        return pipeline_transcript(run)
    return agentic_transcript(run)
