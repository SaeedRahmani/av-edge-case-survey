"""
figures.py
----------
Canonical scientometric figure generator, used for BOTH the paper (survey
corpus) and the living repository (full bibliography). Run it once per corpus:

  # paper figures (survey corpus only)
  python figures.py --input ../../analysis/corpus_classified.csv \
                    --outdir ../../Figures --trend-max 2025

  # living-repo figures (seed + discovered)
  python figures.py --input ../data/bibliography.csv \
                    --outdir ../figures --trend-max 2026

Outputs (PDF + PNG):
  scientometric_trend.*   publications per year, stacked by class
  topic_class_heatmap.*   BERTopic data-driven topics x survey class
  keyword_network.*       curated concept co-occurrence network
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from classify import embed

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "mathtext.default": "regular",
        "axes.linewidth": 0.6,
    }
)

# 4 categories. Assessment is a ground-truth-only seed category; the automated
# classifier targets the other three.
CLASSES = ["Perception-related", "Trajectory-related", "Knowledge-driven", "Assessment"]
CLASS_COLORS = {
    "Perception-related": "#2f6fd0",
    "Trajectory-related": "#2ca25f",
    "Knowledge-driven": "#e08214",
    "Assessment": "#8c6bb1",
}
# short legend names
CLASS_SHORT = {"Perception-related": "Perception", "Trajectory-related": "Trajectory",
               "Knowledge-driven": "Knowledge", "Assessment": "Assessment"}

# Standard domain keywords (label -> regex). These are recognised AV edge-case
# terms; the NODES, their SIZE, and the EDGES are all derived from how often the
# terms actually occur (and co-occur) in the titles + abstracts of the corpus,
# i.e. they are data-driven keyword occurrences, not hand-assigned concepts.
# Only terms that occur in at least `min_count` papers are shown.
CONCEPTS = [
    ("anomaly\ndetection", r"anomal"),
    ("out-of-distribution\ndetection", r"out[\s\-]of[\s\-]distribution|\bood\b"),
    ("novelty\ndetection", r"novelty"),
    ("outlier\ndetection", r"outlier"),
    ("semantic\nsegmentation", r"semantic segmentation|\bsegmentation\b"),
    ("object\ndetection", r"object detection"),
    ("uncertainty\nestimation", r"uncertaint|\bconfidence\b|bayesian|calibrat"),
    ("autoencoder /\nreconstruction", r"autoencoder|reconstruction|variational"),
    ("generative\nmodel", r"\bgan\b|generative|diffusion model"),
    ("deep learning", r"deep learning"),
    ("neural\nnetwork", r"neural network|\bcnn\b|\bdnn\b"),
    ("foundation /\nVL model", r"foundation model|vision.?language|large language model|\bllm\b"),
    ("runtime\nmonitoring", r"runtime monitor|online monitor|\bmonitoring\b"),
    ("corner /\nedge case", r"corner[\s\-]case|edge[\s\-]case"),
    ("scenario\ngeneration", r"scenario generation|scenario[\s\-]based|generat\w+ scenario"),
    ("safety-critical\nscenario", r"safety[\s\-]critical|critical scenario"),
    ("surrogate\nsafety / TTC", r"surrogate|time[\s\-]to[\s\-]collision|\bttc\b|post[\s\-]encroachment|traffic conflict"),
    ("near-miss", r"near[\s\-]miss"),
    ("importance\nsampling", r"importance sampling|rare event|risk estimation"),
    ("reinforcement\nlearning", r"reinforcement learning"),
    ("adversarial /\nfalsification", r"adversarial|falsif|search[\s\-]based test"),
    ("simulation /\nsim-to-real", r"simulation|sim[\s\-]?to[\s\-]?real|\bcarla\b"),
    ("verification &\nvalidation", r"verification|\bvalidation\b"),
    ("naturalistic\ndriving", r"naturalistic"),
    ("ontology /\nknowledge graph", r"ontolog|knowledge graph|knowledge[\s\-]based"),
    ("operational\ndesign domain", r"operational design domain|\bodd\b"),
    ("lidar", r"\blidar\b"),
    ("radar", r"\bradar\b"),
    ("camera", r"\bcamera\b"),
    ("point cloud", r"point cloud"),
    ("benchmark /\ndataset", r"\bbenchmark\b|\bdataset\b"),
]


def load(path):
    """Rows used in the category figures: ground-truth seed + discovered only
    (fig_include == 1), with a known category."""
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if str(r.get("fig_include", "")).strip() in ("1", "1.0", "True")
                and r.get("category") in CLASSES]
    return rows


def _save(fig, outdir, name):
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", pdf, "(+ .png)")


# --------------------------------------------------------------------------- #
def fig_trend(rows, outdir, ymin, ymax):
    years = list(range(ymin, ymax + 1))
    counts = {c: [0] * len(years) for c in CLASSES}
    for r in rows:
        try:
            y = int(str(r["year"])[:4])
        except (ValueError, TypeError):
            continue
        if ymin <= y <= ymax and r["category"] in counts:
            counts[r["category"]][y - ymin] += 1

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bottom = np.zeros(len(years))
    for c in CLASSES:
        vals = np.array(counts[c])
        ax.bar(years, vals, bottom=bottom, label=CLASS_SHORT[c], color=CLASS_COLORS[c],
               width=0.82, edgecolor="white", linewidth=0.3)
        bottom += vals
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(years[::2])
    ax.tick_params(length=2)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    _save(fig, outdir, "scientometric_trend")


# --------------------------------------------------------------------------- #
def fig_topic_heatmap(rows, outdir, n_topics=9):
    from bertopic import BERTopic
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [(r["title"] + ". " + (r.get("abstract") or "")).strip() for r in rows]
    embeddings = np.asarray(embed(docs))
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    cluster_model = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    tm = BERTopic(embedding_model=None, hdbscan_model=cluster_model,
                  vectorizer_model=vectorizer, calculate_probabilities=False, verbose=False)
    topics, _ = tm.fit_transform(docs, embeddings=embeddings)

    labels = {t: ", ".join(w for w, _ in tm.get_topic(t)[:3]) for t in set(topics)}
    mat = np.zeros((n_topics, len(CLASSES)))
    tcount = Counter(topics)
    for t, r in zip(topics, rows):
        mat[t, CLASSES.index(r["category"])] += 1
    rs = mat.sum(axis=1, keepdims=True)
    frac = np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)
    order = sorted(range(n_topics), key=lambda t: (frac[t].argmax(), -frac[t].max()))
    frac = frac[order]
    ylabels = [f"{labels[t]}  (n={tcount[t]})" for t in order]

    cmap = LinearSegmentedColormap.from_list("b", ["#f7f7f7", "#08306b"])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([CLASS_SHORT[c] for c in CLASSES], rotation=25, ha="right")
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels(ylabels, fontsize=6.5)
    for i in range(n_topics):
        for j in range(len(CLASSES)):
            v = frac[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "#333333", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("fraction of topic's papers", fontsize=6.5)
    ax.set_title("Data-driven topics vs. survey taxonomy", fontsize=8)
    fig.tight_layout(pad=0.3)
    _save(fig, outdir, "topic_class_heatmap")


# --------------------------------------------------------------------------- #
def fig_concept_network(rows, outdir):
    import networkx as nx

    texts = [((r["title"] + " " + (r.get("abstract") or "")).lower()) for r in rows]
    cls_idx = np.array([CLASSES.index(r["category"]) for r in rows])

    # membership matrix: paper x concept
    M = np.zeros((len(rows), len(CONCEPTS)), dtype=bool)
    for j, (_, pat) in enumerate(CONCEPTS):
        rx = re.compile(pat)
        for i, t in enumerate(texts):
            if rx.search(t):
                M[i, j] = True

    counts = M.sum(axis=0)
    # keep the most frequent real keywords (occurring in >= min_count papers),
    # capped to keep the map legible
    min_count = max(5, int(0.02 * len(rows)))
    cand = [j for j in range(len(CONCEPTS)) if counts[j] >= min_count]
    cand.sort(key=lambda j: -counts[j])
    keep = sorted(cand[:22])
    labels = [CONCEPTS[j][0] for j in keep]
    Mk = M[:, keep]
    counts = counts[keep]

    # dominant class per concept
    node_class = []
    for j in range(len(keep)):
        cc = Counter(cls_idx[Mk[:, j]])
        node_class.append(CLASSES[cc.most_common(1)[0][0]])

    co = Mk.T.astype(int) @ Mk.astype(int)  # concept co-occurrence (papers sharing both)
    n = len(keep)

    # Build a k-nearest-neighbour keyword graph: every term is linked to its few
    # strongest co-occurring partners. This keeps the graph connected regardless
    # of corpus size (an absolute threshold fragments the smaller paper corpus).
    min_co = 2
    topk = 4
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a in range(n):
        partners = sorted(((int(co[a, b]), b) for b in range(n)
                           if b != a and co[a, b] >= min_co), reverse=True)[:topk]
        for w, b in partners:
            if G.has_edge(a, b):
                continue
            G.add_edge(a, b, weight=w)
    G.remove_nodes_from([j for j in list(G.nodes) if G.degree(j) == 0])
    # keep only the largest connected component so the layout does not scatter
    if G.number_of_nodes() and not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
    nodes = list(G.nodes)

    pos = nx.spring_layout(G, k=3.4 / np.sqrt(max(len(nodes), 1)), seed=5,
                           weight="weight", iterations=1500)

    # size by frequency but cap so the largest hub does not dominate
    capped = {j: min(counts[j], int(0.35 * len(rows))) for j in nodes}
    node_sizes = [150 + 130 * np.sqrt(capped[j]) for j in nodes]
    node_colors = [CLASS_COLORS[node_class[j]] for j in nodes]
    ew = [G[u][v]["weight"] for u, v in G.edges]
    ewmax = max(ew) if ew else 1
    widths = [0.3 + 2.2 * (w / ewmax) for w in ew]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#9fb3c8", alpha=0.45,
                           connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors,
                           node_size=node_sizes, linewidths=0.6, edgecolors="white",
                           alpha=0.95, ax=ax)
    for j in nodes:
        ax.text(pos[j][0], pos[j][1], labels[j], ha="center", va="center",
                fontsize=6.0, color="black", zorder=5,
                linespacing=0.9)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                          markersize=8, label=CLASS_SHORT[cls])
               for cls, col in CLASS_COLORS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
              ncol=4, bbox_to_anchor=(0.5, -0.04))
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    _save(fig, outdir, "keyword_network")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trend-min", type=int, default=2015)
    ap.add_argument("--trend-max", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.input)
    print(f"loaded {len(rows)} classified papers from {args.input}")
    fig_trend(rows, args.outdir, args.trend_min, args.trend_max)
    fig_topic_heatmap(rows, args.outdir)
    fig_concept_network(rows, args.outdir)
    print("done ->", args.outdir)


if __name__ == "__main__":
    main()
