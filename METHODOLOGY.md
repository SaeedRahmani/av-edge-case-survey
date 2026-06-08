# Methodology and Repository Guide

This document contains the detailed workflow, figures, tuning parameters, and reproducibility notes for the **AV Edge-Case Survey** living repository. The [README](README.md) keeps a short introduction first, then mirrors the generated bibliography so visitors can immediately see the paper list.

## How It Works

```text
Papers used in the manuscript
    |
    v
Search OpenAlex for new related papers
    |
    v
Remove duplicates and unsuitable sources
    |
    v
Check whether each paper fits the survey topic
    |
    v
Classify papers by the manuscript structure
    |
    v
Human review for uncertain cases
    |
    v
Update the living bibliography and repository paper list
```

![AV edge-case survey methodology flowchart](figures/methodology_flow_living.png)

OpenAlex returns many loosely related automated-driving papers. To keep the living bibliography close to the paper's scope, each candidate is compared with the papers already used in the manuscript and checked for relevance to automated-driving edge cases. The scored candidate list is preserved in [`data/candidates_scored.csv`](data/candidates_scored.csv).

On the first run, the repository added 120 carefully selected papers to the manuscript paper list. After that, each monthly refresh keeps the existing selected papers and adds at most five new papers, so the living bibliography can grow without becoming unmanageable.

In the most recent run (OpenAlex snapshot through **2026-06-07**): **2,830** records were found; unsuitable sources, paper-owned overlaps, and duplicates were removed; **2,519** records remained after cleaning; **706** passed the relevance check; and **120** selected additions were merged with the **244** manuscript papers into a **364**-study living bibliography. The discovered mix is Trajectory 72 / Perception 35 / Knowledge 13, with year counts 2023:20 / 2024:32 / 2025:47 / 2026:21.

Future monthly runs add at most five more papers that pass the same relevance checks and ranking rules. The added papers are intentionally broader and more recent than the manuscript itself, which is the point of a living companion. Scientometric figures use a separate cutoff and omit 2026, ending at 2025.

## Categories and Assignment

| Category | Meaning |
| --- | --- |
| **Perception-related** | Detecting/generating edge cases in sensing and perception: anomaly, OOD, segmentation, and related methods. |
| **Trajectory-related** | Safety-critical scenarios, scenario generation, surrogate safety, falsification, and trajectory-level evaluation. |
| **Knowledge-driven** | Expert, ontology, and ODD-driven scenario definition and criticality reasoning. |
| **Assessment** | Metrics and evaluation of detection methods; present only for papers already classified in the manuscript. |

Categories are not guessed from hand-written descriptions. Each manuscript paper is labelled by the section of the survey that reviews it. Those labels are stored in the manuscript-paper CSV, [`data/seed_corpus.csv`](data/seed_corpus.csv). New papers are then classified according to the same manuscript structure through [`src/classify.py`](src/classify.py).

On the manuscript paper list, this classifier scores **76% leave-one-out accuracy**: Perception 89%, Trajectory 60%, and Knowledge 76%. *Assessment* overlaps the methods it evaluates, so it is kept as a manuscript-only category and is not an automated target.

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
|- data/                        # paper lists and scored candidate tables
|- figures/                     # living-bibliography figures and PRISMA flow
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
| `SEED_SIM_PCTL` | Minimum similarity to papers already used in the manuscript. | 50 |
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

The default classifier runs without any key: Sentence-BERT, calibrated relevance checks, and nearest-centroid taxonomy. For a higher-accuracy audit pass over borderline (`needs_review`) papers, enable the optional modular LLM layer:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python src/discover.py --no-append
```

Set `use_llm: true` in [`config.yaml`](config.yaml) first. The LLM receives only bibliographic metadata, title, abstract, and the embedding-based screening evidence. It returns structured JSON fields (`llm_relevant`, `llm_class`, `llm_subtopic`, `llm_confidence`, `llm_reason`, plus model/prompt metadata). High-confidence agreement is recorded as `llm_agrees`; high-confidence non-relevance or class disagreement is retained as a human-review flag rather than automatically deleting a paper.

Use `--no-append` when you only want to refresh scores or LLM audit fields for the current bibliography. Omit it for the scheduled monthly update that may append up to five new papers.

## Figures

[`src/figures.py`](src/figures.py) regenerates the scientometric figures for the living bibliography, manuscript papers plus added papers, into [`figures/`](figures): a publication-per-year trend, a BERTopic topic-by-class heatmap, a curated keyword co-occurrence network, and the PRISMA flow. A separate static methodology flowchart in [`figures/methodology_flow_living.tex`](figures/methodology_flow_living.tex) summarizes the monthly update workflow.

The bibliography may include 2026 records, but all scientometric panels generated with `--trend-max 2025` exclude records after 2025. These are the repository's own figures and are deliberately distinct from the figures in the paper, which characterize the smaller, fixed manuscript paper list.

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
