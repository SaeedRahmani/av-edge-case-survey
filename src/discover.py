"""
discover.py
-----------
End-to-end living literature discovery:

  1. harvest recent candidates from OpenAlex (cached);
  2. embed candidates + the expert-curated seed corpus;
  3. de-duplicate candidates against the seed and drop preprint-mill sources;
    4. classify each candidate into the survey's 3 automated categories with the centroid
     classifier (classify.py), whose centroids are the survey's own
     ground-truth-labelled papers;
  5. apply two calibrated gates (relevance margin + seed similarity) as a
     RELEVANCE FLOOR, both calibrated from the seed corpus by leave-one-out;
    6. select the living set by a relevance-led score with a modest recency weight and a SOFT
     PER-CLASS CEILING (no class exceeds `CEILING_FRAC` of the picks), so the
     mix reflects the real field without one class flooding it;
  7. write discovered.csv, candidates_scored.csv and the merged bibliography.csv
     (seed rows carry their GROUND-TRUTH category; discovered rows the predicted
     one). Only ground-truth seed + discovered rows are marked fig_include=1.

Run `python discover.py --tune` to print the calibration table instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import date

import numpy as np
import yaml

import classify as C
import harvest as H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SEED_CSV = os.path.join(DATA, "seed_corpus.csv")
HARVEST_CACHE = os.path.join(DATA, "_harvest_cache.json")
CONFIG_YAML = os.path.join(os.path.dirname(HERE), "config.yaml")


def _load_config() -> dict:
    try:
        with open(CONFIG_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_CFG = _load_config()

SEED_SIM_PCTL  = _CFG.get("seed_sim_percentile", 50)
MARGIN_PCTL    = _CFG.get("margin_percentile", 25)
TOTAL_TARGET   = _CFG.get("total_target", 120)
CEILING_FRAC   = _CFG.get("ceiling_frac", 0.60)
YEAR_CEILING_FRAC = _CFG.get("year_ceiling_frac", 0.0)
RECENCY_WEIGHT = _CFG.get("recency_weight", 0.0)
KNN            = _CFG.get("knn", 5)
FROM_YEAR      = _CFG.get("from_year", 2023)
MAX_PUBLICATION_DATE = _CFG.get("max_publication_date") or date.today().isoformat()
USE_LLM        = _CFG.get("use_llm", False)
LLM_MODEL      = _CFG.get("llm_model", "claude-opus-4-7")
EXCLUDE_SOURCE_PATTERNS = tuple(p.lower() for p in _CFG.get("exclude_source_patterns", []))
EXCLUDE_TITLE_PATTERNS = tuple(p.lower() for p in _CFG.get("exclude_title_patterns", []))

LLM_COLS = ["llm_relevant", "llm_class", "llm_subtopic", "llm_confidence", "llm_reason"]

# Dedicated preprint mills (no peer review) -> excluded. arXiv is kept.
PREPRINT_MILL = ("10.5281/zenodo", "10.20944/preprints", "10.33774/coe",
                 "10.59324/ejaset", "10.31224", "10.21203/rs",
                 "10.22541", "10.36227")


def is_preprint_mill(doi: str) -> bool:
    d = (doi or "").lower()
    return any(p in d for p in PREPRINT_MILL)


def _clean_doi(doi: str) -> str:
    return (doi or "").strip().lower().replace("https://doi.org/", "")


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _canonical_id(row: dict) -> str:
    title_key = _norm_title(row.get("title", ""))
    if title_key:
        return "title:" + title_key
    doi = _clean_doi(row.get("doi", ""))
    if doi:
        return "doi:" + doi
    openalex_id = (row.get("openalex_id") or "").strip().lower()
    if openalex_id:
        return "oa:" + openalex_id
    return ""


def _is_arxiv_version(row: dict) -> bool:
    doi = _clean_doi(row.get("doi", ""))
    venue = (row.get("venue") or "").lower()
    return doi.startswith("10.48550/arxiv") or "arxiv" in venue


def _record_quality(row: dict):
    return (
        int(not _is_arxiv_version(row)),
        int(bool(row.get("abstract"))),
        int(bool(row.get("doi"))),
        int(bool(row.get("openalex_id"))),
        _as_float(row, "cited_by"),
        len(row.get("title") or ""),
    )


def _year(r):
    try:
        return int(str(r.get("year"))[:4])
    except (ValueError, TypeError):
        return 0


def _as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (ValueError, TypeError):
        return 0.0


def _publication_date_ok(row: dict) -> bool:
    publication_date = str(row.get("publication_date") or "").strip()
    max_date = str(MAX_PUBLICATION_DATE)
    if publication_date:
        return publication_date <= max_date
    try:
        return _year(row) <= int(max_date[:4])
    except (ValueError, TypeError):
        return True


def _source_excluded(row: dict) -> bool:
    venue = (row.get("venue") or "").lower()
    title = (row.get("title") or "").lower()
    return any(pattern in venue for pattern in EXCLUDE_SOURCE_PATTERNS) or any(
        pattern in title for pattern in EXCLUDE_TITLE_PATTERNS
    )


def load_seed():
    with open(SEED_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: dict[str, dict] = {}
    for row in rows:
        key = _canonical_id(row)
        if not key or row.get("duplicate_of"):
            continue
        previous = seen.get(key)
        if previous is None or _record_quality(row) > _record_quality(previous):
            seen[key] = row
    return list(seen.values())


def get_candidates(refresh=False):
    if os.path.exists(HARVEST_CACHE) and not refresh:
        with open(HARVEST_CACHE, encoding="utf-8") as f:
            return json.load(f)
    cands = H.harvest(from_year=FROM_YEAR, until_date=MAX_PUBLICATION_DATE)
    with open(HARVEST_CACHE, "w", encoding="utf-8") as f:
        json.dump(cands, f)
    return cands


def doc_text(r):
    return (r.get("title", "") + ". " + (r.get("abstract", "") or "")).strip()


def knn_sim(query_vecs, ref_vecs, k, exclude_self=False):
    sims = query_vecs @ ref_vecs.T
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


DISC_COLS = ["canonical_id", "openalex_id", "doi", "title", "year", "publication_date",
             "venue", "work_type", "cited_by", "category", "class_score", "margin",
             "seed_sim", "rank_score", "needs_review", "review_reason", "source",
             "matched_query"]
BIB_COLS = ["canonical_id", "openalex_id", "doi", "title", "year", "publication_date",
            "venue", "work_type", "cited_by", "category", "label_source", "fig_include",
            "class_score", "margin", "seed_sim", "needs_review", "review_reason",
            "source", "abstract"]


def review_reason(row: dict, seed_threshold: float, margin_threshold: float) -> str:
    reasons = []
    if _as_float(row, "seed_sim") < seed_threshold + 0.03:
        reasons.append("near seed-sim floor")
    if _as_float(row, "margin") < margin_threshold + 0.02:
        reasons.append("near relevance floor")
    venue = (row.get("venue") or "").lower()
    title = (row.get("title") or "").lower()
    current_year = date.today().year
    if _year(row) == current_year and int(_as_float(row, "cited_by")) == 0 and "arxiv" in venue:
        reasons.append("current-year zero-citation preprint")
    if _year(row) == current_year and not row.get("doi"):
        reasons.append("current-year missing DOI")
    if "student abstract" in title or "workshop" in title:
        reasons.append("short/workshop item")
    return "; ".join(reasons)


def rank_key(row: dict):
    return (
        -_as_float(row, "rank_score"),
        -_as_float(row, "margin"),
        -_as_float(row, "cited_by"),
        _year(row),
        row.get("title", ""),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    seed = load_seed()
    seed_keys = {_canonical_id(row) for row in seed if _canonical_id(row)}
    seed_titles = {_norm_title(row["title"]) for row in seed}

    print("Embedding seed corpus ...")
    seed_vecs = C.embed([doc_text(r) for r in seed])
    seed_loo_sim = knn_sim(seed_vecs, seed_vecs, KNN, exclude_self=True)
    seed_margin = np.array([r["margin"] for r in C.classify(seed)])
    s_thresh = float(np.percentile(seed_loo_sim, SEED_SIM_PCTL))
    m_thresh = float(np.percentile(seed_margin, MARGIN_PCTL))
    print(f"  floor: seed-sim p{SEED_SIM_PCTL}={s_thresh:.3f}, margin p{MARGIN_PCTL}={m_thresh:.3f}")

    cands = get_candidates(refresh=args.refresh)
    n_mill = 0
    n_future = 0
    n_source = 0
    new_by_key: dict[str, dict] = {}
    for candidate in cands:
        title = candidate.get("title", "")
        norm_title = _norm_title(title)
        candidate["canonical_id"] = _canonical_id(candidate)
        if (candidate["canonical_id"] in seed_keys or norm_title in seed_titles
                or len(title) <= 10):
            continue
        if is_preprint_mill(candidate.get("doi", "")):
            n_mill += 1
            continue
        if not _publication_date_ok(candidate):
            n_future += 1
            continue
        if _source_excluded(candidate):
            n_source += 1
            continue
        previous = new_by_key.get(candidate["canonical_id"])
        if previous is None or _record_quality(candidate) > _record_quality(previous):
            new_by_key[candidate["canonical_id"]] = candidate
    new = list(new_by_key.values())
    print(f"Candidates: {len(cands)} harvested, dropped {n_mill} preprint-mill, "
          f"{n_future} future-dated, {n_source} source/title exclusions, "
          f"{len(new)} after dedup")

    cand_vecs = C.embed([doc_text(r) for r in new])
    cand_seed_sim = knn_sim(cand_vecs, seed_vecs, KNN)
    cand_cls = C.classify(new)
    cand_margin = np.array([r["margin"] for r in cand_cls])

    if args.tune:
        print(f"{'pctl':>5}{'sim':>8}{'pass':>8}")
        for p in (30, 40, 50, 60, 70):
            st = np.percentile(seed_loo_sim, p)
            b = int(((cand_seed_sim >= st) & (cand_margin >= m_thresh)).sum())
            print(f"{p:>5}{st:>8.3f}{b:>8}")
        return

    # relevance floor
    passing = []
    for index, candidate in enumerate(cand_cls):
        if cand_seed_sim[index] >= s_thresh and cand_margin[index] >= m_thresh:
            rr = dict(candidate)
            rr["canonical_id"] = _canonical_id(rr)
            rr["category"] = rr.pop("class")
            rr["seed_sim"] = round(float(cand_seed_sim[index]), 4)
            rr["rank_score"] = round(
                rr["seed_sim"] + RECENCY_WEIGHT * max(0, _year(rr) - FROM_YEAR), 4
            )
            rr["source"] = "auto-discovered"
            rr["label_source"] = "predicted"
            rr["fig_include"] = 1
            rr["review_reason"] = review_reason(rr, s_thresh, m_thresh)
            rr["needs_review"] = bool(rr["review_reason"])
            passing.append(rr)
    passing.sort(key=rank_key)

    with open(os.path.join(DATA, "candidates_scored.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DISC_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(passing)

    # soft per-class ceiling selection (natural proportions, no class dominates)
    ceiling = int(CEILING_FRAC * TOTAL_TARGET)
    year_ceiling = int(YEAR_CEILING_FRAC * TOTAL_TARGET) if YEAR_CEILING_FRAC else 0
    per = Counter()
    per_year = Counter()
    keep = []
    for r in passing:
        if len(keep) >= TOTAL_TARGET:
            break
        if per[r["category"]] >= ceiling:
            continue
        row_year = str(r.get("year", ""))[:4]
        if year_ceiling and per_year[row_year] >= year_ceiling:
            continue
        keep.append(r)
        per[r["category"]] += 1
        per_year[row_year] += 1
    keep.sort(key=lambda r: (r["category"], -r["rank_score"]))

    # optional LLM second-opinion pass on borderline papers (use_llm in config.yaml)
    if USE_LLM:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Warning: use_llm=true in config but ANTHROPIC_API_KEY is not set; skipping LLM pass.")
        else:
            import classify_llm as LLM
            flagged = [r for r in keep if r.get("needs_review")]
            if flagged:
                print(f"Running LLM second-opinion on {len(flagged)} borderline papers ...")
                reviewed = {r["openalex_id"]: r
                            for r in LLM.classify_llm(flagged, model=LLM_MODEL)}
                for r in keep:
                    if r["openalex_id"] in reviewed:
                        lr = reviewed[r["openalex_id"]]
                        r["llm_relevant"]   = lr.get("llm_relevant")
                        r["llm_class"]      = lr.get("llm_class")
                        r["llm_confidence"] = lr.get("llm_confidence")
                        r["llm_reason"]     = lr.get("llm_reason")
                        if lr.get("llm_class") in C.CLASSES:
                            r["category"] = lr["llm_class"]
                print(f"  LLM review complete ({len(flagged)} papers)")

    disc_cols = DISC_COLS + (LLM_COLS if (USE_LLM and os.getenv("ANTHROPIC_API_KEY")) else [])
    with open(os.path.join(DATA, "discovered.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=disc_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(keep)

    # merged bibliography: seed (ground-truth category) + discovered (predicted)
    merged = []
    for r in seed:
        d = dict(r)
        d["canonical_id"] = _canonical_id(d)
        d["source"] = "seed"
        d["needs_review"] = False
        d["review_reason"] = ""
        d["seed_sim"] = ""
        d["class_score"] = ""
        d["margin"] = ""
        merged.append(d)
    merged.extend(keep)
    with open(os.path.join(DATA, "bibliography.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BIB_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(merged)

        print(f"\n{len(passing)} above floor; CURATED {len(keep)} "
            f"(ceiling {ceiling}/class, {year_ceiling or 'off'}/year):")
    for c, n in Counter(r["category"] for r in keep).most_common():
        print(f"  {c:20s} {n}")
    print(f"  by year: {dict(sorted(Counter(str(r['year'])[:4] for r in keep).items()))}")
    print(f"  flagged for review: {sum(1 for r in keep if r['needs_review'])}")
    print(f"  bibliography.csv total: {len(merged)} "
          f"(fig_include: {sum(int(r.get('fig_include',0) or 0) for r in merged)})")


if __name__ == "__main__":
    main()
