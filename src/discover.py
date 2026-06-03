"""
discover.py
-----------
End-to-end living literature discovery:

  1. harvest recent candidates from OpenAlex (cached);
  2. embed candidates + the expert-curated seed corpus;
  3. de-duplicate candidates against the seed and drop preprint-mill sources;
  4. classify each candidate into the survey's 4 categories with the centroid
     classifier (classify.py), whose centroids are the survey's own
     ground-truth-labelled papers;
  5. apply two calibrated gates (relevance margin + seed similarity) as a
     RELEVANCE FLOOR, both calibrated from the seed corpus by leave-one-out;
  6. select the living set by a blended relevance+recency score with a SOFT
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

import numpy as np

import classify as C
import harvest as H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SEED_CSV = os.path.join(DATA, "seed_corpus.csv")
HARVEST_CACHE = os.path.join(DATA, "_harvest_cache.json")

SEED_SIM_PCTL = 50      # relevance floor (b): >= median seed centrality
MARGIN_PCTL = 25        # relevance floor (a): >= 25th-pctl seed margin
TOTAL_TARGET = 120      # size of the curated living addition (~1.5x seed)
CEILING_FRAC = 0.60     # soft ceiling: no class may exceed 60% of the picks
RECENCY_WEIGHT = 0.05   # recency bonus per year after FROM_YEAR in the rank score
KNN = 5
FROM_YEAR = 2023

# Dedicated preprint mills (no peer review) -> excluded. arXiv is kept.
PREPRINT_MILL = ("10.5281/zenodo", "10.20944/preprints", "10.33774/coe",
                 "10.59324/ejaset", "10.31224", "10.21203/rs",
                 "10.22541", "10.36227")


def is_preprint_mill(doi: str) -> bool:
    d = (doi or "").lower()
    return any(p in d for p in PREPRINT_MILL)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _year(r):
    try:
        return int(str(r.get("year"))[:4])
    except (ValueError, TypeError):
        return 0


def load_seed():
    with open(SEED_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_candidates(refresh=False):
    if os.path.exists(HARVEST_CACHE) and not refresh:
        with open(HARVEST_CACHE, encoding="utf-8") as f:
            return json.load(f)
    cands = H.harvest(from_year=FROM_YEAR)
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


DISC_COLS = ["openalex_id", "doi", "title", "year", "venue", "cited_by",
             "category", "class_score", "margin", "seed_sim", "rank_score",
             "needs_review", "source", "matched_query"]
BIB_COLS = ["openalex_id", "doi", "title", "year", "venue", "cited_by",
            "category", "label_source", "fig_include", "class_score", "margin",
            "seed_sim", "needs_review", "source", "abstract"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    seed = load_seed()
    seed_titles = {_norm_title(r["title"]) for r in seed}
    seed_dois = {r["doi"] for r in seed if r.get("doi")}
    seed_oa = {r.get("openalex_id", "") for r in seed if r.get("openalex_id")}

    print("Embedding seed corpus ...")
    seed_vecs = C.embed([doc_text(r) for r in seed])
    seed_loo_sim = knn_sim(seed_vecs, seed_vecs, KNN, exclude_self=True)
    seed_margin = np.array([r["margin"] for r in C.classify(seed)])
    s_thresh = float(np.percentile(seed_loo_sim, SEED_SIM_PCTL))
    m_thresh = float(np.percentile(seed_margin, MARGIN_PCTL))
    print(f"  floor: seed-sim p{SEED_SIM_PCTL}={s_thresh:.3f}, margin p{MARGIN_PCTL}={m_thresh:.3f}")

    cands = get_candidates(refresh=args.refresh)
    n_mill = 0
    new_by_title: dict[str, dict] = {}
    for c in cands:
        nt = _norm_title(c.get("title", ""))
        if (c.get("openalex_id") in seed_oa or (c.get("doi") and c["doi"] in seed_dois)
                or nt in seed_titles or len(c.get("title", "")) <= 10):
            continue
        if is_preprint_mill(c.get("doi", "")):
            n_mill += 1
            continue
        prev = new_by_title.get(nt)
        if prev is None or len(c.get("abstract") or "") > len(prev.get("abstract") or ""):
            new_by_title[nt] = c
    new = list(new_by_title.values())
    print(f"Candidates: {len(cands)} harvested, dropped {n_mill} preprint-mill, "
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
    for i, r in enumerate(cand_cls):
        if cand_seed_sim[i] >= s_thresh and cand_margin[i] >= m_thresh:
            rr = dict(r)
            rr["category"] = rr.pop("class")
            rr["seed_sim"] = round(float(cand_seed_sim[i]), 4)
            rr["rank_score"] = round(rr["seed_sim"] + RECENCY_WEIGHT * max(0, _year(rr) - FROM_YEAR), 4)
            rr["source"] = "auto-discovered"
            rr["fig_include"] = 1
            rr["needs_review"] = bool(cand_seed_sim[i] < s_thresh + 0.03
                                      or cand_margin[i] < m_thresh + 0.02)
            passing.append(rr)
    passing.sort(key=lambda r: -r["rank_score"])

    with open(os.path.join(DATA, "candidates_scored.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DISC_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(passing)

    # soft per-class ceiling selection (natural proportions, no class dominates)
    ceiling = int(CEILING_FRAC * TOTAL_TARGET)
    per = Counter()
    keep = []
    for r in passing:
        if len(keep) >= TOTAL_TARGET:
            break
        if per[r["category"]] >= ceiling:
            continue
        keep.append(r)
        per[r["category"]] += 1
    keep.sort(key=lambda r: (r["category"], -r["rank_score"]))

    with open(os.path.join(DATA, "discovered.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DISC_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(keep)

    # merged bibliography: seed (ground-truth category) + discovered (predicted)
    merged = []
    for r in seed:
        d = dict(r)
        d["source"] = "seed"
        d["needs_review"] = False
        d["seed_sim"] = ""
        d["class_score"] = ""
        d["margin"] = ""
        merged.append(d)
    merged.extend(keep)
    with open(os.path.join(DATA, "bibliography.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BIB_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(merged)

    print(f"\n{len(passing)} above floor; CURATED {len(keep)} "
          f"(ceiling {ceiling}/class):")
    for c, n in Counter(r["category"] for r in keep).most_common():
        print(f"  {c:20s} {n}")
    print(f"  by year: {dict(sorted(Counter(str(r['year'])[:4] for r in keep).items()))}")
    print(f"  flagged for review: {sum(1 for r in keep if r['needs_review'])}")
    print(f"  bibliography.csv total: {len(merged)} "
          f"(fig_include: {sum(int(r.get('fig_include',0) or 0) for r in merged)})")


if __name__ == "__main__":
    main()
