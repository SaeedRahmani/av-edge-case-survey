"""
discover.py
-----------
End-to-end living literature discovery:

  1. harvest recent candidates from OpenAlex (cached);
  2. embed candidates + the expert-curated seed corpus;
  3. de-duplicate candidates against the seed;
  4. classify each candidate into the survey taxonomy (perception / trajectory /
     knowledge-driven + subsection);
  5. apply TWO calibrated gates so the discovered set stays consistent with the
     survey's scope and does not flood the bibliography with off-topic work:
       (a) relevance margin  = sim(best on-topic anchor) - sim(best off-topic anchor)
       (b) seed similarity   = mean cosine to the candidate's k nearest seed papers
     Both thresholds are CALIBRATED FROM THE SEED CORPUS itself via leave-one-out,
     so a discovered paper must be at least as on-topic as a chosen percentile of
     the papers the authors already cite.
  6. write data/discovered.csv and the merged data/bibliography.csv.

Run `python discover.py --tune` to print the calibration table instead of writing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

import classify as C  # shared taxonomy + embedding model
import harvest as H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SEED_CSV = os.path.join(DATA, "seed_corpus.csv")
HARVEST_CACHE = os.path.join(DATA, "_harvest_cache.json")

# Calibration percentiles (of the seed's own leave-one-out distribution).
# A candidate must be at least as on-topic as the seed paper at this percentile.
# These act as a RELEVANCE FLOOR; the curated set is then ranked and capped per
# class so the living bibliography stays small and high-precision rather than
# echoing the whole (very large) field.
SEED_SIM_PCTL = 50      # gate (b): >= median seed centrality
MARGIN_PCTL = 25        # gate (a): >= 25th-pctl seed relevance margin
PER_CLASS_CAP = 60      # curated picks per top-level class
RECENCY_WEIGHT = 0.05   # bonus per year after FROM_YEAR added to the rank score
KNN = 5                 # neighbours used for seed-similarity
FROM_YEAR = 2023        # surface genuinely new work
# The living bibliography is meant to be ~1.4-1.5x the survey corpus and to
# emphasise the MOST RECENT literature, so within each class we rank candidates
# recency-first (then by relevance) before applying the per-class cap.


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


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
    """mean cosine to k nearest ref vectors (all normalized)."""
    sims = query_vecs @ ref_vecs.T
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    part = np.sort(sims, axis=1)[:, -k:]
    return part.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="print calibration table only")
    ap.add_argument("--refresh", action="store_true", help="re-query OpenAlex")
    ap.add_argument("--seed-pctl", type=float, default=SEED_SIM_PCTL)
    ap.add_argument("--margin-pctl", type=float, default=MARGIN_PCTL)
    args = ap.parse_args()

    seed = load_seed()
    seed_titles = {_norm_title(r["title"]) for r in seed}
    seed_dois = {r["doi"] for r in seed if r.get("doi")}
    seed_oa = {r.get("openalex_id", "") for r in seed if r.get("openalex_id")}

    print("Embedding seed corpus ...")
    seed_vecs = np.asarray(C.embed([doc_text(r) for r in seed]))

    # --- calibrate thresholds from the seed itself (leave-one-out) ---
    seed_loo_sim = knn_sim(seed_vecs, seed_vecs, KNN, exclude_self=True)
    seed_classified = C.classify(seed)
    seed_margin = np.array([r["margin"] for r in seed_classified])

    s_thresh = float(np.percentile(seed_loo_sim, args.seed_pctl))
    m_thresh = float(np.percentile(seed_margin, args.margin_pctl))
    print(f"  seed kNN-sim  p{args.seed_pctl:.0f} = {s_thresh:.3f} "
          f"(seed range {seed_loo_sim.min():.2f}..{seed_loo_sim.max():.2f})")
    print(f"  seed margin   p{args.margin_pctl:.0f} = {m_thresh:.3f}")

    # --- candidates ---
    cands = get_candidates(refresh=args.refresh)
    # drop ones already in the seed, and collapse near-duplicate records
    # (preprint + published share a normalized title) keeping the richer one
    new_by_title: dict[str, dict] = {}
    for c in cands:
        nt = _norm_title(c.get("title", ""))
        if (
            c.get("openalex_id") in seed_oa
            or (c.get("doi") and c["doi"] in seed_dois)
            or nt in seed_titles
            or len(c.get("title", "")) <= 10
        ):
            continue
        prev = new_by_title.get(nt)
        if prev is None or len(c.get("abstract") or "") > len(prev.get("abstract") or ""):
            new_by_title[nt] = c
    new = list(new_by_title.values())
    print(f"\nCandidates: {len(cands)} harvested, {len(new)} after dedup "
          f"(vs seed + near-duplicate titles)")

    cand_vecs = np.asarray(C.embed([doc_text(r) for r in new]))
    cand_seed_sim = knn_sim(cand_vecs, seed_vecs, KNN)
    cand_classified = C.classify(new)
    cand_margin = np.array([r["margin"] for r in cand_classified])

    if args.tune:
        print("\n=== Calibration: candidates passing at various seed-sim percentiles ===")
        print(f"{'pctl':>5} {'sim_thr':>8} {'pass(sim)':>10} {'pass(sim&margin)':>17}")
        for p in (20, 30, 35, 40, 50, 60, 70):
            st = np.percentile(seed_loo_sim, p)
            a = int((cand_seed_sim >= st).sum())
            b = int(((cand_seed_sim >= st) & (cand_margin >= m_thresh)).sum())
            print(f"{p:>5} {st:>8.3f} {a:>10} {b:>17}")
        # show what gets in / out at the default threshold
        keep_mask = (cand_seed_sim >= s_thresh) & (cand_margin >= m_thresh)
        order = np.argsort(-cand_seed_sim)
        print(f"\n--- TOP 25 kept (sim, margin, class) at p{args.seed_pctl:.0f} ---")
        shown = 0
        for i in order:
            if keep_mask[i]:
                r = cand_classified[i]
                print(f"  {cand_seed_sim[i]:.3f} m={cand_margin[i]:+.2f} "
                      f"[{r['class'][:10]:10s}] {r['title'][:66]}")
                shown += 1
                if shown >= 25:
                    break
        print("\n--- 12 BORDERLINE just-below the gate (would be excluded) ---")
        below = [i for i in order if not keep_mask[i]
                 and cand_seed_sim[i] < s_thresh][:12]
        for i in sorted(below, key=lambda i: -cand_seed_sim[i]):
            print(f"  {cand_seed_sim[i]:.3f} m={cand_margin[i]:+.2f} "
                  f"{cand_classified[i]['title'][:66]}")
        return

    # --- relevance floor: candidates that pass both gates ---
    passing = []
    for i, r in enumerate(cand_classified):
        if (cand_seed_sim[i] >= s_thresh) and (cand_margin[i] >= m_thresh):
            rr = dict(r)
            rr["seed_sim"] = round(float(cand_seed_sim[i]), 4)
            rr["source"] = "auto-discovered"
            rr["needs_review"] = bool(
                cand_seed_sim[i] < s_thresh + 0.03 or cand_margin[i] < m_thresh + 0.02
            )
            passing.append(rr)
    def _year(r):
        try:
            return int(str(r.get("year"))[:4])
        except (ValueError, TypeError):
            return 0

    # Blend relevance with a recency bonus so the living bibliography emphasises
    # the most recent literature WITHOUT sacrificing relevance.
    for r in passing:
        r["rank_score"] = r["seed_sim"] + RECENCY_WEIGHT * max(0, _year(r) - FROM_YEAR)
    passing.sort(key=lambda r: -r["rank_score"])

    out_cols = ["openalex_id", "doi", "title", "year", "venue", "cited_by",
                "class", "subsection", "class_score", "margin", "seed_sim",
                "needs_review", "source", "matched_query"]

    # full transparent scored list (everything above the floor)
    with open(os.path.join(DATA, "candidates_scored.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(passing)

    # --- curated set: top-N per class, ranked by seed similarity ---
    from collections import Counter, defaultdict

    per_class = defaultdict(list)
    for r in passing:
        if len(per_class[r["class"]]) < PER_CLASS_CAP:
            per_class[r["class"]].append(r)
    keep = [r for items in per_class.values() for r in items]
    keep.sort(key=lambda r: (r["class"], -r["seed_sim"]))

    with open(os.path.join(DATA, "discovered.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    # merged bibliography (seed + discovered)
    merged = []
    for r in seed_classified:
        d = dict(r)
        d["source"] = "seed"
        d["needs_review"] = False
        d["seed_sim"] = ""
        merged.append(d)
    merged.extend(keep)
    with open(os.path.join(DATA, "bibliography.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols + ["abstract"], extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    print(f"\n{len(passing)} candidates above relevance floor "
          f"(sim>={s_thresh:.3f}, margin>={m_thresh:.3f})")
    print(f"CURATED {len(keep)} new papers (top {PER_CLASS_CAP}/class):")
    for c, n in Counter(r["class"] for r in keep).most_common():
        print(f"  {c:20s} {n}")
    nrev = sum(1 for r in keep if r["needs_review"])
    print(f"  flagged for human review: {nrev}")
    print(f"  wrote data/discovered.csv (curated), data/candidates_scored.csv "
          f"({len(passing)} full), data/bibliography.csv ({len(merged)} total)")


if __name__ == "__main__":
    main()
