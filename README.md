# Living Survey: Edge-Case Detection & Assessment for Automated Driving

A **living, automatically updated companion** to the survey
*"Edge Cases in Automated Driving: A Survey of Detection and Assessment Methods"*.

This repository operationalises the survey's methodology as a reproducible,
semi-automated pipeline. It (1) carries the **expert-curated reference corpus**
of the paper as a frozen *seed*, (2) **discovers new papers** from
[OpenAlex](https://openalex.org) on a schedule, (3) **classifies** every paper
into the survey's taxonomy (*perception-related* / *trajectory-related* /
*knowledge-driven*, and their subsections), and (4) keeps a tightly-screened,
human-reviewable [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) up to date.

The goal is **objectivity, scalability, and replicability**: the literature base
of the survey can be re-derived and extended by anyone, with the expert
synthesis preserved as the seed and the automation acting as a transparent
augmentation — *not* a replacement.

---

## How it works

```
OpenAlex API ──▶ harvest ──▶ de-duplicate ──▶ embed (Sentence-BERT)
                                                     │
   seed corpus (263 expert-curated papers) ─────────┤
                                                     ▼
                       classify (nearest-centroid, trained on
                        the survey's own section labels)
                                                     ▼
                            ┌──────── two calibrated screening gates ────────┐
                            │ (a) topical relevance margin                    │
                            │     = sim(class centroid) − sim(off-topic)      │
                            │ (b) seed similarity                             │
                            │     = mean cosine to k nearest seed papers      │
                            │ thresholds calibrated from the seed itself      │
                            │ (leave-one-out percentiles)                     │
                            └─────────────────────────────────────────────────┘
                                                     ▼
            rank by relevance+recency, soft per-class ceiling ──▶ human review
                                                     ▼
                    discovered.csv · candidates_scored.csv · BIBLIOGRAPHY.md
```

**Why two gates and calibration?** Naively searching OpenAlex returns thousands
of loosely-related AV papers. To stay *consistent with the survey's actual
scope* and avoid flooding the bibliography, a candidate must be (a) topically on
the edge-case theme and (b) at least as close to the existing corpus as a
*median paper the authors already cite*. Both thresholds are derived from the
seed corpus by leave-one-out, so the screening is principled and reproducible
rather than hand-picked. The curated set is then ranked by a blended
relevance+recency score under a **soft per-class ceiling** (no class exceeds 60%
of the picks); the full scored list is preserved in `candidates_scored.csv`.

In the most recent run: **2,836** records harvested → **69** preprint-mill
sources dropped → **2,533** after de-duplication → **578** above the relevance
floor → **120** curated additions (soft per-class ceiling, ranked by a blended
relevance+recency score), merged with the **263**-paper seed into a **383**-study
living bibliography (~1.5× the survey corpus, dominated by 2024–2026 work). The
discovered mix (Trajectory 72 / Perception 36 / Knowledge 12) reflects the real
recent literature. The discovered set is intentionally broader and more recent
than the survey itself, which is the point of a *living* companion.

---

## Categories & how they are assigned

| Category | Meaning |
|---|---|
| **Perception-related** | detecting/generating edge cases in sensing & perception (anomaly, OOD, segmentation, …) |
| **Trajectory-related** | safety-critical scenarios, scenario generation, surrogate safety, falsification, … |
| **Knowledge-driven** | expert/ontology/ODD-driven scenario definition & criticality reasoning |
| **Assessment** | metrics & evaluation of detection methods (ground-truth seed category only) |

Categories are **not** guessed from hand-written descriptions. Each *seed* paper
is labelled by the **section of the survey that reviews it** (ground truth; see
[`analysis/section_labels.py`](../analysis/section_labels.py)). The class
prototypes (centroids) of those ground-truth papers then classify newly
*discovered* papers ([`src/classify.py`](src/classify.py)). On the seed this
classifier scores **75% leave-one-out accuracy** (Perception 90% / Trajectory
61% / Knowledge 70%). *Assessment* overlaps the methods it evaluates, so it is
kept as a ground-truth-only seed category and is not an automated target.

---

## Repository layout

```
living-survey/
├── README.md
├── BIBLIOGRAPHY.md            # auto-generated, browsable, grouped by category
├── config.yaml               # search terms + tuning parameters (mirrors src defaults)
├── requirements.txt
├── src/
│   ├── harvest.py            # OpenAlex retrieval (Boolean phrase queries)
│   ├── classify.py          # centroid classifier (trained on the seed's section labels)
│   ├── classify_llm.py      # OPTIONAL modular-LLM second opinion (needs API key)
│   ├── discover.py          # end-to-end: harvest → screen → classify → select → write
│   ├── figures.py           # scientometric figures (trend, heatmap, network)
│   └── report.py            # build BIBLIOGRAPHY.md from the data
├── data/
│   ├── seed_corpus.csv       # 263 seed papers + ground-truth category & label_source
│   ├── discovered.csv        # curated new papers (soft per-class ceiling)
│   ├── candidates_scored.csv # full scored candidate list (transparency)
│   └── bibliography.csv      # merged seed + discovered, categorised
├── figures/                  # scientometric figures for the LIVING corpus (+ PRISMA)
└── .github/workflows/update.yml   # monthly auto-refresh
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python discover.py            # harvest, screen, classify, write data/*.csv
python report.py              # regenerate BIBLIOGRAPHY.md

# inspect the calibration without writing anything:
python discover.py --tune
# force a fresh OpenAlex query (ignore the local harvest cache):
python discover.py --refresh
```

### Tuning

Strictness is controlled in `src/discover.py` (mirrored in `config.yaml`):

| Parameter | Meaning | Default |
|---|---|---|
| `SEED_SIM_PCTL` | seed-similarity floor (percentile of seed LOO distribution) | 50 |
| `MARGIN_PCTL` | topical-relevance floor | 25 |
| `TOTAL_TARGET` | size of the curated living addition | 120 |
| `CEILING_FRAC` | soft ceiling: max share of picks from one class | 0.60 |
| `RECENCY_WEIGHT` | recency bonus per year added to the rank score | 0.05 |
| `FROM_YEAR` | earliest publication year to consider | 2023 |

Raise the percentiles / lower the cap for a stricter, smaller set.

### Optional: modular-LLM second opinion

The default classifier already runs without any key (Sentence-BERT). For a
higher-accuracy pass over the borderline (`needs_review`) papers:

```bash
export ANTHROPIC_API_KEY=sk-...
python src/classify_llm.py
```

---

## Figures

`src/figures.py` regenerates the scientometric figures for **the living corpus**
(seed + discovered) into [`figures/`](figures): a publication-per-year trend (which
rises through 2024–2026 as new work is discovered), a BERTopic topic × class
heatmap, a curated keyword co-occurrence network, and the PRISMA flow. These are
the repository's *own* figures and are deliberately distinct from the figures in
the paper, which characterise the smaller, fixed survey corpus:

```bash
python src/figures.py --input data/bibliography.csv --outdir figures --trend-max 2026
```

| Publication trend | Topics × taxonomy | Concept co-occurrence |
|:--:|:--:|:--:|
| ![trend](figures/scientometric_trend.png) | ![heatmap](figures/topic_class_heatmap.png) | ![network](figures/keyword_network.png) |

![PRISMA](figures/prisma_flow.png)

## Reproducibility & archival

The pipeline is deterministic given a fixed OpenAlex snapshot date. For a
citable, frozen version (as referenced in the paper), archive a release on
[Zenodo](https://zenodo.org) to obtain a DOI:

1. Tag a release on GitHub (`v1.0.0`).
2. Enable the repository in Zenodo; a DOI is minted automatically.
3. Cite that DOI in the paper's data-availability statement.

---

## How to cite

If you use this resource, please cite the survey (and, if relevant, the Zenodo
snapshot DOI). The pipeline builds on OpenAlex, Sentence-BERT
(Reimers & Gurevych, 2019), and BERTopic (Grootendorst, 2022).

## License

MIT — see [LICENSE](LICENSE). Bibliographic metadata is sourced from OpenAlex
under CC0.
