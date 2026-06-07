"""
harvest.py
----------
Retrieve candidate papers from the OpenAlex API using the same Boolean search
logic described in the survey, expressed as a set of phrase queries over title
and abstract.  Results are unioned and de-duplicated by OpenAlex id.

OpenAlex is free and requires no API key.  We use the "polite pool" by passing
a mailto.  Full-boolean parsing is not supported by the API, so the survey's
search string is decomposed into AV-term x edge-case-term phrase queries.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.path.join(os.path.dirname(HERE), "config.yaml")


def _load_config() -> dict:
    try:
        with open(CONFIG_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_CFG = _load_config()

MAILTO = _CFG.get("mailto", "saeedrmd@gmail.com")
MAX_PUBLICATION_DATE = _CFG.get("max_publication_date") or date.today().isoformat()

# Edge-case / corner-case concept phrases (the survey's second AND-block).
EDGE_TERMS = _CFG.get("edge_terms") or [
    "edge case",
    "corner case",
    "anomaly detection",
    "out-of-distribution",
    "novelty detection",
    "rare event",
    "safety-critical scenario",
    "critical scenario",
    "scenario generation",
    "near-miss",
    "surrogate safety",
]

# Automated-driving context phrases (the survey's first AND-block).
AV_TERMS = _CFG.get("av_terms") or [
    "autonomous driving",
    "automated driving",
    "autonomous vehicle",
    "automated vehicle",
    "self-driving",
]


def _get(url: str, mailto: str = MAILTO) -> dict:
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}mailto={mailto}"
    req = urllib.request.Request(url, headers={"User-Agent": f"av-edge-case-survey ({mailto})"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def reconstruct_abstract(inv_index) -> str:
    if not inv_index:
        return ""
    pos = []
    for word, idxs in inv_index.items():
        for i in idxs:
            pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def harvest(from_year: int = 2023, until_date: str | None = None,
            per_query: int = 60, max_pages: int = 2) -> list[dict]:
    """Return de-duplicated candidate works published in the configured date window."""
    seen: dict[str, dict] = {}
    until_date = until_date or MAX_PUBLICATION_DATE
    n_queries = 0
    for av in AV_TERMS:
        for edge in EDGE_TERMS:
            n_queries += 1
            phrase = f'"{edge}" "{av}"'
            q = urllib.parse.quote(phrase)
            cursor = "*"
            fetched = 0
            for _ in range(max_pages):
                filters = [f"from_publication_date:{from_year}-01-01", "type:article|preprint"]
                if until_date:
                    filters.append(f"to_publication_date:{until_date}")
                url = (
                    "https://api.openalex.org/works"
                    f"?search={q}"
                    f"&filter={','.join(filters)}"
                    f"&per-page=50&cursor={cursor}"
                )
                try:
                    data = _get(url)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! query failed ({phrase}): {e}")
                    break
                results = data.get("results", [])
                for w in results:
                    oid = (w.get("id") or "").replace("https://openalex.org/", "")
                    if not oid or oid in seen:
                        continue
                    seen[oid] = {
                        "openalex_id": oid,
                        "title": (w.get("title") or "").strip(),
                        "year": w.get("publication_year", ""),
                        "publication_date": w.get("publication_date", ""),
                        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                        "venue": (
                            (w.get("primary_location") or {}).get("source") or {}
                        ).get("display_name", "")
                        if w.get("primary_location")
                        else "",
                        "work_type": w.get("type", ""),
                        "cited_by": w.get("cited_by_count", 0),
                        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                        "matched_query": phrase,
                    }
                    fetched += 1
                cursor = (data.get("meta") or {}).get("next_cursor")
                if not cursor or len(results) < 50 or fetched >= per_query:
                    break
                time.sleep(0.1)
            time.sleep(0.08)
        print(f"  harvested {len(seen)} unique candidates from {n_queries} phrase queries "
                    f"({from_year}-01-01 to {until_date})")
    return list(seen.values())


if __name__ == "__main__":
    cands = harvest()
    print(f"total: {len(cands)}")
