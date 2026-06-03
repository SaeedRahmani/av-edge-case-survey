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

CLASSES = ["Perception-related", "Trajectory-related", "Knowledge-driven"]
CLASS_COLORS = {
    "Perception-related": "#2f6fd0",
    "Trajectory-related": "#2ca25f",
    "Knowledge-driven": "#e08214",
}

# Curated domain concepts -> regex over (title + abstract). Node size = number
# of papers matching; node colour = dominant class of those papers. Multi-line
# labels (\n) keep nodes compact, as in a concept map.
CONCEPTS = [
    # perception
    ("out-of-distribution\ndetection", r"out[- ]of[- ]distribution|\bood\b"),
    ("anomaly detection", r"anomal"),
    ("autoencoder\nVAE", r"autoencoder|variational|\bvae\b"),
    ("semantic\nsegmentation", r"segmentation"),
    ("novelty detection", r"novelty"),
    ("uncertainty\nconfidence", r"uncertaint|confidence|bayesian|calibrat"),
    ("dataset\nbenchmark", r"dataset|benchmark"),
    ("runtime\nmonitoring", r"runtime monitor|online monitor|\bmonitoring\b"),
    ("neural networks\ndeep learning", r"deep learning|neural network|convolutional|\bcnn\b"),
    ("transfer &\nfew-shot learning", r"transfer learning|few[- ]shot|zero[- ]shot"),
    ("foundation models\nLLM", r"foundation model|vision.?language|large language model|\bllm\b"),
    ("feature &\nlatent space", r"latent space|feature space|embedding space|representation learning"),
    # trajectory
    ("scenario\ngeneration", r"scenario generation|scenario-based|generat\w+ scenario"),
    ("safety-critical\nscenarios", r"safety[- ]critical|critical scenario"),
    ("surrogate safety\nmeasures", r"surrogate|time[- ]to[- ]collision|\bttc\b|post[- ]encroachment|traffic conflict"),
    ("adversarial &\nfalsification", r"adversarial|falsif|search[- ]based test"),
    ("importance sampling\nrare event", r"importance sampling|rare[- ]event|risk estimation"),
    ("trajectory &\nmotion planning", r"trajectory predict|motion planning|motion predict|path planning"),
    ("reinforcement\nlearning", r"reinforcement learning|\bdrl\b"),
    ("simulation\nsim2real", r"simulation|sim2real|sim-to-real|\bcarla\b"),
    ("testing &\nverification", r"\btesting\b|verification|\bvalidation\b"),
    # knowledge-driven
    ("ontology &\nknowledge graph", r"ontolog|knowledge graph|knowledge[- ]based"),
    ("influencing factors\n& ODD", r"influencing factor|operational design domain|\bodd\b|expert knowledge"),
    ("crash &\naccident data", r"crash|accident|naturalistic driving"),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("class") in CLASSES]
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
        if ymin <= y <= ymax and r["class"] in counts:
            counts[r["class"]][y - ymin] += 1

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bottom = np.zeros(len(years))
    for c in CLASSES:
        vals = np.array(counts[c])
        ax.bar(years, vals, bottom=bottom, label=c, color=CLASS_COLORS[c],
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
        mat[t, CLASSES.index(r["class"])] += 1
    rs = mat.sum(axis=1, keepdims=True)
    frac = np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)
    order = sorted(range(n_topics), key=lambda t: (frac[t].argmax(), -frac[t].max()))
    frac = frac[order]
    ylabels = [f"{labels[t]}  (n={tcount[t]})" for t in order]

    cmap = LinearSegmentedColormap.from_list("b", ["#f7f7f7", "#08306b"])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(["Perception", "Trajectory", "Knowledge"], rotation=20, ha="right")
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
    cls_idx = np.array([CLASSES.index(r["class"]) for r in rows])

    # membership matrix: paper x concept
    M = np.zeros((len(rows), len(CONCEPTS)), dtype=bool)
    for j, (_, pat) in enumerate(CONCEPTS):
        rx = re.compile(pat)
        for i, t in enumerate(texts):
            if rx.search(t):
                M[i, j] = True

    counts = M.sum(axis=0)
    keep = [j for j in range(len(CONCEPTS)) if counts[j] >= 3]
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
    edge_thr = max(4, int(0.02 * len(rows)))

    G = nx.Graph()
    for j in range(n):
        G.add_node(j)
    for a in range(n):
        for b in range(a + 1, n):
            if co[a, b] >= edge_thr:
                G.add_edge(a, b, weight=int(co[a, b]))
    G.remove_nodes_from([j for j in list(G.nodes) if G.degree(j) == 0])
    nodes = list(G.nodes)

    pos = nx.spring_layout(G, k=2.6 / np.sqrt(max(len(nodes), 1)), seed=11,
                           weight="weight", iterations=600)

    cmax = counts.max() if len(counts) else 1
    node_sizes = [260 + 240 * np.sqrt(counts[j]) for j in nodes]
    node_colors = [CLASS_COLORS[node_class[j]] for j in nodes]
    ew = [G[u][v]["weight"] for u, v in G.edges]
    ewmax = max(ew) if ew else 1
    widths = [0.3 + 2.2 * (w / ewmax) for w in ew]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#9fb3c8", alpha=0.5,
                           connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors,
                           node_size=node_sizes, linewidths=0.6, edgecolors="white",
                           alpha=0.95, ax=ax)
    for j in nodes:
        ax.text(pos[j][0], pos[j][1], labels[j], ha="center", va="center",
                fontsize=6.6, color="black", zorder=5,
                linespacing=0.95)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                          markersize=8, label=cls) for cls, col in CLASS_COLORS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
              ncol=3, bbox_to_anchor=(0.5, -0.04))
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
