"""
lib_classify.py
---------------
Shared, dependency-light classification logic used by BOTH the paper-side
analysis and the living-survey repository.

Approach (transparent + reproducible):
  * embed each paper (title + abstract) with a sentence-transformer
    (all-MiniLM-L6-v2, the de-facto standard small model);
  * embed a set of *class anchors* whose wording is taken directly from the
    survey's own taxonomy (top-level classes + subsections);
  * assign each paper to the nearest class/subsection by cosine similarity
    (nearest-centroid over the anchor prototypes).

The anchor text is intentionally verbatim-aligned with the paper so the
data-driven assignment *recovers the manual taxonomy* rather than inventing a
new one.  A relevance gate (max anchor similarity) flags off-topic papers for
human review.
"""
from __future__ import annotations

import functools

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Top-level classes -> subsections, each with an anchor description drawn from
# the survey text. The first sentence names the class; the rest are cue phrases.
TAXONOMY = {
    "Perception-related": {
        "Reconstructive and Generative": (
            "Perception-related edge cases detected by reconstructive and "
            "generative models: autoencoders, GANs, VAEs, normalizing flows, "
            "image reconstruction error, out-of-distribution and novelty "
            "detection in camera or LiDAR data."
        ),
        "Probability and Confidence Scores": (
            "Perception edge cases detected via probability-based methods and "
            "confidence scores: softmax uncertainty, Bayesian deep learning, "
            "Monte Carlo dropout, calibration, epistemic uncertainty in object "
            "detection and semantic segmentation."
        ),
        "Feature and Activation Extraction": (
            "Perception edge cases detected by feature and activation-value "
            "extraction: deep feature embeddings, latent space distance, "
            "activation patterns of neural networks for anomaly and corner "
            "case detection in images."
        ),
        "Foundation Model and Adaptive Learning": (
            "Perception edge cases detected with foundation models, vision "
            "language models, large pretrained models, zero-shot and adaptive "
            "or continual learning for novel object and scene understanding."
        ),
        "Other Perception Methods": (
            "Other perception-related corner case detection methods including "
            "reconstruction-free, prediction-based, optical flow, or "
            "multi-frame temporal consistency checks on sensor data."
        ),
    },
    "Trajectory-related": {
        "Surrogate Safety Metrics": (
            "Trajectory-related edge cases identified with surrogate safety "
            "measures: time-to-collision, post-encroachment time, brake threat "
            "number, conflict indicators, near-miss and traffic conflict "
            "metrics."
        ),
        "Probability Estimation": (
            "Trajectory-related safety-critical scenarios identified via "
            "probability estimation and risk: importance sampling, rare-event "
            "probability, statistical risk estimation of collisions and "
            "criticality."
        ),
        "Machine Learning (trajectory)": (
            "Trajectory and motion edge cases detected with machine learning: "
            "trajectory prediction, reinforcement learning, anomaly detection "
            "in driving behavior, motion planning under uncertainty."
        ),
        "Challenging the System Under Test": (
            "Safety-critical scenario generation by challenging the system "
            "under test: adversarial agents, falsification, search-based "
            "testing, scenario optimization to provoke failures of the "
            "autonomous driving stack."
        ),
        "Novel Scenario Generation": (
            "Generation of novel and critical driving scenarios: naturalistic "
            "driving data, scenario sampling, simulation-based scenario "
            "generation, parameterized concrete scenarios for testing."
        ),
    },
    "Knowledge-driven": {
        "Influencing Factors": (
            "Knowledge-driven edge cases defined from influencing factors and "
            "expert knowledge: operational design domain attributes, ontology "
            "of driving environment, taxonomy of influencing conditions."
        ),
        "Formalisation of Description": (
            "Formalisation of edge case description using expert rules, "
            "ontologies, knowledge graphs, scenario description languages and "
            "formal logic to specify challenging situations."
        ),
        "Qualification and Classification": (
            "Knowledge-driven qualification and classification of scenarios by "
            "criticality and relevance using heuristic multi-criteria expert "
            "rules and risk reasoning."
        ),
    },
}

# A neutral / off-topic anchor set: if a paper is closest to one of these,
# it is likely NOT an AV edge-case paper and should be flagged for review.
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
    return model.encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64
    )


def _flat_anchors():
    """Return (labels, vectors) for every subsection anchor + offtopic anchors."""
    labels, texts = [], []
    for cls, subs in TAXONOMY.items():
        for sub, desc in subs.items():
            labels.append((cls, sub))
            texts.append(desc)
    off_start = len(labels)
    for t in OFFTOPIC_ANCHORS:
        labels.append(("__OFFTOPIC__", "__OFFTOPIC__"))
        texts.append(t)
    return labels, np.asarray(embed(texts)), off_start


def classify(records, relevance_threshold=0.0):
    """
    records: list of dicts each with 'title' and (optional) 'abstract'.
    Adds keys: class, subsection, class_score (cosine to best on-topic anchor),
               offtopic_score, relevant (bool), margin (best on-topic minus best off-topic).

    relevance_threshold gates on (best_ontopic - best_offtopic) margin.
    """
    labels, anchor_vecs, off_start = _flat_anchors()
    texts = [
        (r.get("title", "") + ". " + (r.get("abstract", "") or "")).strip()
        for r in records
    ]
    doc_vecs = np.asarray(embed(texts))
    sims = doc_vecs @ anchor_vecs.T  # cosine (vectors are normalized)

    on = sims[:, :off_start]
    off = sims[:, off_start:]
    best_on_idx = on.argmax(axis=1)
    best_on = on.max(axis=1)
    best_off = off.max(axis=1)
    margin = best_on - best_off

    out = []
    for i, r in enumerate(records):
        cls, sub = labels[best_on_idx[i]]
        rr = dict(r)
        rr["class"] = cls
        rr["subsection"] = sub
        rr["class_score"] = round(float(best_on[i]), 4)
        rr["offtopic_score"] = round(float(best_off[i]), 4)
        rr["margin"] = round(float(margin[i]), 4)
        rr["relevant"] = bool(margin[i] >= relevance_threshold)
        out.append(rr)
    return out
