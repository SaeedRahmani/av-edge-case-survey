"""
classify.py
-----------
Shared classifier used by BOTH the paper-side analysis and the av-edge-case-survey
repository.

Unlike the earlier version (which compared papers to hand-written anchor
sentences), categories are now learnt from GROUND TRUTH: the centroids of the
seed papers whose category is known from the section of the survey that cites
them (see analysis/section_labels.py). A paper is assigned to the nearest class
centroid; a small off-topic anchor set flags non-AV papers.

Automated classes (3): Perception-related, Trajectory-related, Knowledge-driven.
Assessment is retained as a ground-truth-only figure category.

The labelled seed lives in data/seed_corpus.csv (column `category`, with
`label_source == groundtruth` for the section-cited papers used to build the
centroids).
"""
from __future__ import annotations

import csv
import functools
import os
import re

import numpy as np


def _load_config() -> dict:
    try:
        import yaml

        config_path = os.path.join(os.path.dirname(HERE), "config.yaml")
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}


HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = _load_config()

MODEL_NAME = _CFG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
# The automated classifier targets the three DETECTION classes only. "Assessment"
# is a cross-cutting category that overlaps the methods it evaluates, so as a
# prediction target it steals papers and is unreliable (43% LOO recall). It is
# therefore kept as a ground-truth-only seed category for the figures (see
# FIGURE_CATEGORIES) but is NOT a centroid the classifier can assign to.
CLASSES = ["Perception-related", "Trajectory-related", "Knowledge-driven"]
FIGURE_CATEGORIES = CLASSES + ["Assessment"]

# Taxonomy with subtopics consumed by classify_llm.py for optional LLM prompting.
TAXONOMY = {
    "Perception-related": {
        "anomaly detection": "detecting anomalies in sensor data",
        "out-of-distribution detection": "identifying inputs outside training distribution",
        "novelty detection": "detecting novel or unseen scenarios in perception",
        "uncertainty estimation": "quantifying model prediction uncertainty",
        "object detection": "detecting objects under safety-critical conditions",
        "semantic segmentation": "scene understanding under edge conditions",
    },
    "Trajectory-related": {
        "scenario generation": "generating safety-critical test scenarios",
        "safety-critical scenario": "scenarios with high collision or failure risk",
        "surrogate safety measure": "metrics such as TTC and PRIT for risk assessment",
        "trajectory prediction": "predicting vehicle or pedestrian trajectories",
        "reinforcement learning": "RL-based testing or adversarial scenario discovery",
        "falsification": "finding inputs that falsify system specifications",
        "simulation-based testing": "testing AV systems in simulation environments",
    },
    "Knowledge-driven": {
        "ontology-based approach": "expert knowledge encoded in ontologies",
        "knowledge graph": "structured knowledge representations for AV safety",
        "operational design domain": "ODD definition and boundary detection",
        "rule-based criticality": "rule or logic-based edge case detection",
    },
}

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


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _canonical_id(row: dict) -> str:
    title_key = _norm_title(row.get("title", ""))
    if title_key:
        return "title:" + title_key
    doi = (row.get("doi") or "").strip().lower().replace("https://doi.org/", "")
    if doi:
        return "doi:" + doi
    openalex_id = (row.get("openalex_id") or "").strip().lower()
    if openalex_id:
        return "oa:" + openalex_id
    return ""


def _is_figure_row(row: dict) -> bool:
    return str(row.get("fig_include", "1")).strip() in ("1", "1.0", "True", "true")


def _training_rows(rows: list[dict]) -> list[dict]:
    train = [row for row in rows
             if row.get("label_source") == "groundtruth"
             and row.get("category") in CLASSES
             and _is_figure_row(row)]
    seen: set[str] = set()
    unique = []
    for row in train:
        key = _canonical_id(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


@functools.lru_cache(maxsize=1)
def _centroids_and_offtopic(seed_csv=SEED_CSV):
    """Build per-class centroids from the ground-truth-labelled seed papers."""
    with open(seed_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = _training_rows(rows)
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
    cls_sims = vecs @ cmat.T          # N x 3
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
        rows = _training_rows(list(csv.DictReader(f)))
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
