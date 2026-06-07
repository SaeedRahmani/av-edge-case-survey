# Methodology and Repository Guide

This document contains the detailed workflow, figures, tuning parameters, and reproducibility notes for the **AV Edge-Case Survey** living repository. The [README](README.md) keeps a short introduction first, then mirrors the generated bibliography so visitors can immediately see the paper list.

## How It Works

```text
OpenAlex API -> harvest -> de-duplicate -> embed (Sentence-BERT)
                                                    |
   seed corpus (263 source records; 244 canonical works) --|
                                                    v
                      classify (nearest-centroid, trained on
                       the survey's own section labels)
                                                    v
                           +------ two calibrated screening gates ------+
                           | (a) topical relevance margin               |
                           |     = sim(class centroid) - sim(off-topic) |
                           | (b) seed similarity                        |
                           |     = mean cosine to k nearest seed papers |
                           | thresholds calibrated from the seed itself |
                           | (leave-one-out percentiles)                |
                           +--------------------------------------------+
                                                    v
           rank by relevance-led score, append <=5/month under class/year ceilings
                                                    v
                   discovered.csv - candidates_scored.csv - BIBLIOGRAPHY.md
```

![AV edge-case survey methodology flowchart](figures/methodology_flow_living.png)

**Why two gates and calibration?** Naively searching OpenAlex returns thousands of loosely related AV papers. To stay consistent with the survey's actual scope and avoid flooding the bibliography, a candidate must be (a) topically on the edge-case theme and (b) at least as close to the existing corpus as a median paper the authors already cite. Both thresholds are derived from the seed corpus by leave-one-out, so the screening is principled and reproducible rather than hand-picked.

The candidate list is ranked by a relevance-led score with a modest recency nudge (`recency_weight: 0.01` by default). On the first run, the pipeline bootstraps the curated living addition to 120 papers. After that, each monthly refresh retains the existing selected papers and appends at most five new papers, still under soft per-class and per-year ceilings. The full scored list is preserved in [`data/candidates_scored.csv`](data/candidates_scored.csv).

In the most recent run (OpenAlex snapshot through **2026-06-07**): **2,830** records were harvested; **68** preprint-mill sources, **1** configured source/title exclusion, and **36** paper-owned seed/reference overlaps were dropped; **206** duplicate or near-duplicate records were also removed; **2,519** records remained after de-duplication; **706** passed the relevance floor; and **120** bootstrap curated additions were merged with the **244**-work canonical seed into a **364**-study living bibliography. The discovered mix is Trajectory 72 / Perception 35 / Knowledge 13, with year counts 2023:20 / 2024:32 / 2025:47 / 2026:21.

Future monthly runs add at most five more papers that pass the same gates and ranking rules. The discovered set is intentionally broader and more recent than the survey itself, which is the point of a living companion. Scientometric figures use a separate cutoff and omit 2026, ending at 2025.

## Categories and Assignment

| Category | Meaning |
| --- | --- |
| **Perception-related** | Detecting/generating edge cases in sensing and perception: anomaly, OOD, segmentation, and related methods. |
| **Trajectory-related** | Safety-critical scenarios, scenario generation, surrogate safety, falsification, and trajectory-level evaluation. |
| **Knowledge-driven** | Expert, ontology, and ODD-driven scenario definition and criticality reasoning. |
| **Assessment** | Metrics and evaluation of detection methods; ground-truth seed category only. |

Categories are not guessed from hand-written descriptions. Each seed paper is labelled by the uncommented citation in the section of the survey that reviews it in the source manuscript. Those ground-truth labels are stored in [`data/seed_corpus.csv`](data/seed_corpus.csv). The class prototypes (centroids) of those ground-truth papers then classify newly discovered papers through [`src/classify.py`](src/classify.py).

On the seed corpus, this classifier scores **76% leave-one-out accuracy**: Perception 89%, Trajectory 60%, and Knowledge 76%. *Assessment* overlaps the methods it evaluates, so it is kept as a ground-truth-only seed category and is not an automated target.

## Repository Layout

```text
av-edge-case-survey/
|- README.md                    # short front page + generated bibliography mirror
|- METHODOLOGY.md               # detailed workflow, figures, tuning, reproducibility
|- BIBLIOGRAPHY.md              # auto-generated, browsable, grouped by category
|- config.yaml                  # search terms + tuning parameters
|- requirements.txt
|- src/
|  |- harvest.py                # OpenAlex retrieval
|  |- classify.py               # centroid classifier
|  |- classify_llm.py           # optional modular-LLM second opinion
|  |- discover.py               # harvest, screen, classify, select, write
|  |- figures.py                # scientometric figures
|  `- report.py                 # builds BIBLIOGRAPHY.md and updates README.md
|- data/
|  |- seed_corpus.csv           # 244 canonical seed works + labels
|  |- discovered.csv            # curated living additions
|  |- candidates_scored.csv     # full scored candidate list
|  `- bibliography.csv          # merged seed + discovered corpus
|- figures/                     # living-corpus figures and PRISMA flow
`- .github/workflows/update.yml # monthly auto-refresh
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python discover.py            # harvest, screen, classify, write data/*.csv
python report.py              # regenerate BIBLIOGRAPHY.md and README.md bibliography section

# inspect the calibration without writing anything:
python discover.py --tune

# force a fresh OpenAlex query, ignoring the local harvest cache:
python discover.py --refresh
```

## Tuning

Strictness is controlled in [`src/discover.py`](src/discover.py) and mirrored in [`config.yaml`](config.yaml).

| Parameter | Meaning | Default |
| --- | --- | --- |
| `SEED_SIM_PCTL` | Seed-similarity floor, percentile of seed leave-one-out distribution. | 50 |
| `MARGIN_PCTL` | Topical-relevance floor. | 25 |
| `BOOTSTRAP_TARGET` | First-run size of the curated living addition when `discovered.csv` is empty. | 120 |
| `MONTHLY_ADD_LIMIT` | Maximum number of newly selected papers appended by each later run. | 5 |
| `CEILING_FRAC` | Soft ceiling: maximum share of picks from one class. | 0.60 |
| `YEAR_CEILING_FRAC` | Optional maximum share of picks from one publication year. | 0.40 |
| `RECENCY_WEIGHT` | Optional recency bonus per year added to the rank score. | 0.01 |
| `MAX_PUBLICATION_DATE` | Latest publication date included in the bibliography snapshot; `null` means the run date. | null |
| `FROM_YEAR` | Earliest publication year to consider. | 2023 |

Raise the percentiles for a stricter set, or lower `MONTHLY_ADD_LIMIT` for slower growth.

## Optional Modular-LLM Second Opinion

The default classifier runs without any key: Sentence-BERT, calibrated gates, and nearest-centroid taxonomy. For a higher-accuracy audit pass over borderline (`needs_review`) papers, enable the optional modular LLM layer:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python src/discover.py --no-append
```

Set `use_llm: true` in [`config.yaml`](config.yaml) first. The LLM receives only bibliographic metadata, title, abstract, and the embedding-based screening evidence. It returns structured JSON fields (`llm_relevant`, `llm_class`, `llm_subtopic`, `llm_confidence`, `llm_reason`, plus model/prompt metadata). High-confidence agreement is recorded as `llm_agrees`; high-confidence non-relevance or class disagreement is retained as a human-review flag rather than automatically deleting a paper.

Use `--no-append` when you only want to refresh scores or LLM audit fields for the current bibliography. Omit it for the scheduled monthly update that may append up to five new papers.

## Figures

[`src/figures.py`](src/figures.py) regenerates the scientometric figures for the living corpus, seed plus discovered papers, into [`figures/`](figures): a publication-per-year trend, a BERTopic topic-by-class heatmap, a curated keyword co-occurrence network, and the PRISMA flow. A separate static methodology flowchart in [`figures/methodology_flow_living.tex`](figures/methodology_flow_living.tex) summarizes the monthly update workflow.

The bibliography may include 2026 records, but all scientometric panels generated with `--trend-max 2025` exclude records after 2025. These are the repository's own figures and are deliberately distinct from the figures in the paper, which characterize the smaller, fixed survey corpus.

```bash
python src/figures.py --input data/bibliography.csv --outdir figures --trend-min 2015 --trend-max 2025
```

| Publication trend | Topics by taxonomy | Concept co-occurrence |
| :---: | :---: | :---: |
| ![trend](figures/scientometric_trend.png) | ![heatmap](figures/topic_class_heatmap.png) | ![network](figures/keyword_network.png) |

![PRISMA](figures/prisma_flow.png)

## Reproducibility and Archival

The living workflow uses the run date by default, so it can keep growing. For a citable, frozen version as referenced in the paper, set `max_publication_date` to a fixed date, run the pipeline, and archive a release on [Zenodo](https://zenodo.org) to obtain a DOI.

1. Tag a release on GitHub, such as `v1.0.0`.
2. Enable the repository in Zenodo; a DOI is minted automatically.
3. Cite that DOI in the paper's data-availability statement.

## How to Cite

If you use this resource, please cite the survey and, if relevant, the Zenodo snapshot DOI. The pipeline builds on OpenAlex, Sentence-BERT (Reimers and Gurevych, 2019), and BERTopic (Grootendorst, 2022).

## License

MIT. See [LICENSE](LICENSE). Bibliographic metadata is sourced from OpenAlex under CC0.
