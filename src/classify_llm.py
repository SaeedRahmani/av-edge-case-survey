"""
classify_llm.py  (OPTIONAL)
---------------------------
Drop-in LLM-based relevance + taxonomy classifier, used only when a higher-
accuracy, modular-LLM pass is desired (the reviewer's "modular LLM" suggestion).

It is NOT required: the default pipeline in discover.py uses transformer
sentence-embeddings (a BERT-family model), which already classifies the corpus
without any API key.  This module adds an optional second opinion for the
borderline papers flagged `needs_review` by the embedding gate.

Enable by setting `use_llm: true` in config.yaml and ANTHROPIC_API_KEY, or run
this file directly on data/discovered.csv.
"""
from __future__ import annotations

import json
import os

from classify import TAXONOMY

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


def _prompt(title, abstract):
    taxo = "\n".join(
        f"- {cls}: {', '.join(subtopics)}" for cls, subtopics in _SUBTOPICS.items()
    )
    return (
        f"Taxonomy (class: indicative subtopics):\n{taxo}\n\n"
        f"Paper title: {title}\nAbstract: {abstract[:1200]}\n\n"
        "Respond with ONLY a JSON object: "
        '{"relevant": bool, "class": one of '
        f"{_CLASS_LIST}, "
        '"subtopic": string or null, "confidence": 0-1, "reason": short string}'
    )


def classify_llm(records, model="claude-opus-4-7"):
    """records: list of dicts with title/abstract. Adds llm_* fields. Needs SDK+key."""
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    out = []
    for r in records:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": _prompt(r.get("title", ""), r.get("abstract", "") or "")}],
        )
        text = msg.content[0].text.strip()
        try:
            j = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            j = {"relevant": None, "class": None, "subtopic": None,
                 "confidence": None, "reason": "parse_error"}
        rr = dict(r)
        rr.update({f"llm_{k}": v for k, v in j.items()})
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
