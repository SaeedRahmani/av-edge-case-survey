"""
classify.py
-----------
Shared classifier used by BOTH the paper-side analysis and the living-survey
repository.

Unlike the earlier version (which compared papers to hand-written anchor
sentences), categories are now learnt from GROUND TRUTH: the centroids of the
seed papers whose category is known from the section of the survey that cites
them (see analysis/section_labels.py). A paper is assigned to the nearest class
centroid; a small off-topic anchor set flags non-AV papers.

Classes (4): Perception-related, Trajectory-related, Knowledge-driven, Assessment.

The labelled seed lives in data/seed_corpus.csv (column `category`, with
`label_source == groundtruth` for the section-cited papers used to build the
centroids).
"""
from __future__ import annotations

import csv
import functools
import os

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The automated classifier targets the three DETECTION classes only. "Assessment"
# is a cross-cutting category that overlaps the methods it evaluates, so as a
# prediction target it steals papers and is unreliable (43% LOO recall). It is
# therefore kept as a ground-truth-only seed category for the figures (see
# FIGURE_CATEGORIES) but is NOT a centroid the classifier can assign to.
CLASSES = ["Perception-related", "Trajectory-related", "Knowledge-driven"]
FIGURE_CATEGORIES = CLASSES + ["Assessment"]

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_CSV = os.path.join(os.path.dirname(HERE), "data", "seed_corpus.csv")

# If a paper is closest to one of these it is likely NOT an AV edge-case paper.
OFFTOPIC_ANCHORS = [
    "General machine learning theory unrelated to autonomous driving.",
    "Medical imaging and healthcare diagnosis.",
    "Natural language processing of text documents and chatbots.",
    "Financial markets, economics, and recommender systems.",
    "Cybersecurity intrusion detection and network attacks.",
    "Generic robotics manipulation and industrial control unrelated to driving.",
]


@functools.lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts):
    model = get_model()
    return np.asarray(
        model.encode(list(texts), normalize_embeddings=True,
                     show_progress_bar=False, batch_size=64)
    )


def _doc(r):
    return (r.get("title", "") + ". " + (r.get("abstract", "") or "")).strip()


@functools.lru_cache(maxsize=1)
def _centroids_and_offtopic(seed_csv=SEED_CSV):
    """Build per-class centroids from the ground-truth-labelled seed papers."""
    with open(seed_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows
             if r.get("label_source") == "groundtruth" and r.get("category") in CLASSES]
    if not train:
        raise RuntimeError("no ground-truth training rows found in seed_corpus.csv")
    vecs = embed([_doc(r) for r in train])
    cents = {}
    for c in CLASSES:
        idx = [i for i, r in enumerate(train) if r["category"] == c]
        v = vecs[idx].mean(axis=0)
        cents[c] = v / (np.linalg.norm(v) + 1e-9)
    cmat = np.vstack([cents[c] for c in CLASSES])
    off = embed(OFFTOPIC_ANCHORS)
    return cmat, off


def classify(records, relevance_threshold=0.0):
    """
    records: list of dicts with 'title' (+ optional 'abstract').
    Adds: class, class_score (cosine to nearest class centroid),
          offtopic_score, margin (class_score - best off-topic), relevant.
    """
    cmat, off = _centroids_and_offtopic()
    vecs = embed([_doc(r) for r in records])
    cls_sims = vecs @ cmat.T          # N x 4
    off_sims = vecs @ off.T           # N x K
    best_cls = cls_sims.argmax(axis=1)
    best_cls_sim = cls_sims.max(axis=1)
    best_off = off_sims.max(axis=1)
    margin = best_cls_sim - best_off

    out = []
    for i, r in enumerate(records):
        rr = dict(r)
        rr["class"] = CLASSES[best_cls[i]]
        rr["class_score"] = round(float(best_cls_sim[i]), 4)
        rr["offtopic_score"] = round(float(best_off[i]), 4)
        rr["margin"] = round(float(margin[i]), 4)
        rr["relevant"] = bool(margin[i] >= relevance_threshold)
        out.append(rr)
    return out


def loo_report(seed_csv=SEED_CSV):
    """Leave-one-out accuracy of the centroid classifier on the labelled seed."""
    from collections import Counter

    with open(seed_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("label_source") == "groundtruth" and r.get("category") in CLASSES]
    vecs = embed([_doc(r) for r in rows])
    y = [r["category"] for r in rows]
    ok = 0
    per, perok = Counter(), Counter()
    for i in range(len(rows)):
        cents = {}
        for c in CLASSES:
            idx = [j for j in range(len(rows)) if j != i and y[j] == c]
            if idx:
                v = vecs[idx].mean(0)
                cents[c] = v / (np.linalg.norm(v) + 1e-9)
        pred = max(cents, key=lambda c: vecs[i] @ cents[c])
        per[y[i]] += 1
        if pred == y[i]:
            ok += 1
            perok[y[i]] += 1
    print(f"LOO centroid accuracy: {ok}/{len(rows)} = {ok/len(rows):.0%}")
    for c in CLASSES:
        if per[c]:
            print(f"  {c:20s} recall {perok[c]}/{per[c]} = {perok[c]/per[c]:.0%}")


if __name__ == "__main__":
    loo_report()
