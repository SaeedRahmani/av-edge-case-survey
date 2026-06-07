"""
classify_llm.py  (OPTIONAL)
---------------------------
Drop-in LLM-based relevance + taxonomy classifier, used only when a higher-
accuracy, modular-LLM pass is desired (the reviewer's "modular LLM" suggestion).

It is NOT required: the default pipeline in discover.py uses transformer
sentence-embeddings (a BERT-family model), which already classifies the corpus
without any API key. This module adds an optional second opinion for the
borderline papers flagged `needs_review` by the embedding gate.

Enable by setting `use_llm: true` in config.yaml and ANTHROPIC_API_KEY, or run
this file directly on data/discovered.csv. Outputs are structured audit fields;
final inclusion/exclusion remains a human-review decision.
"""
from __future__ import annotations

import json
import os
from datetime import date

from classify import TAXONOMY

DEFAULT_MODEL = "claude-opus-4-7"
PROMPT_VERSION = "edgecase-screening-v1"
_CLASS_LIST = list(TAXONOMY.keys()) + ["Not relevant"]
_SUBTOPICS = {cls: list(subs.keys()) for cls, subs in TAXONOMY.items()}

SYSTEM = (
    "You are a meticulous research librarian curating a survey on EDGE-CASE / "
    "corner-case detection and assessment for AUTOMATED DRIVING. Decide whether "
    "a paper belongs in the survey and, if so, its top-level class. Be "
    "conservative: papers that are about autonomous driving but NOT about "
    "detecting/generating/assessing edge cases, anomalies, or safety-critical "
    "scenarios are 'Not relevant'."
)


def _short(value, limit=1200):
    return (value or "").replace("\n", " ").strip()[:limit]


def _prompt(record):
    taxo = "\n".join(
        f"- {cls}: {', '.join(subtopics)}" for cls, subtopics in _SUBTOPICS.items()
    )
    current = record.get("category") or record.get("class") or "not assigned"
    evidence = (
        f"Current embedding/centroid class: {current}\n"
        f"Seed similarity: {record.get('seed_sim', '')}\n"
        f"Topical margin: {record.get('margin', '')}\n"
        f"Review flag: {record.get('review_reason', '')}\n"
    )
    return (
        f"Taxonomy (class: indicative subtopics):\n{taxo}\n\n"
        f"Automated screening evidence:\n{evidence}\n"
        f"Paper title: {_short(record.get('title', ''), 300)}\n"
        f"Abstract: {_short(record.get('abstract', ''), 1400)}\n\n"
        "Assess title and abstract conservatively. Respond with ONLY a JSON object: "
        '{"relevant": bool, "class": one of '
        f"{_CLASS_LIST}, "
        '"subtopic": string or null, "confidence": 0-1, "reason": short string}'
    )


def _bool_or_none(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return ""


def _parse_response(text):
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return {
            "relevant": None,
            "class": None,
            "subtopic": None,
            "confidence": None,
            "reason": "parse_error",
        }
    relevant = _bool_or_none(data.get("relevant"))
    paper_class = data.get("class")
    if paper_class not in _CLASS_LIST:
        paper_class = "Not relevant" if relevant is False else None
    return {
        "relevant": relevant,
        "class": paper_class,
        "subtopic": data.get("subtopic"),
        "confidence": _confidence(data.get("confidence")),
        "reason": _short(data.get("reason", ""), 240),
    }


def classify_llm(records, model=DEFAULT_MODEL):
    """records: list of dicts with title/abstract. Adds llm_* fields. Needs SDK+key."""
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    out = []
    reviewed_at = date.today().isoformat()
    for record in records:
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": _prompt(record)}],
        )
        text = msg.content[0].text.strip()
        parsed = _parse_response(text)
        rr = dict(record)
        rr.update({f"llm_{key}": value for key, value in parsed.items()})
        rr["llm_model"] = model
        rr["llm_prompt_version"] = PROMPT_VERSION
        rr["llm_reviewed_at"] = reviewed_at
        out.append(rr)
    return out


if __name__ == "__main__":
    import csv
    import sys

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to use the optional LLM classifier.")
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "data", "discovered.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    flagged = [r for r in rows if r.get("needs_review") == "True"]
    print(f"LLM-reviewing {len(flagged)} borderline papers ...")
    for r in classify_llm(flagged):
        print(f"  rel={r.get('llm_relevant')} ({r.get('llm_confidence')}) "
              f"[{r.get('llm_class')}] {r['title'][:60]} -- {r.get('llm_reason')}")
