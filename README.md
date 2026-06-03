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
                                         classify into taxonomy
                                                     ▼
                            ┌──────── two calibrated screening gates ────────┐
                            │ (a) topical relevance margin                    │
                            │     = sim(best on-topic anchor) − sim(off-topic)│
                            │ (b) seed similarity                             │
                            │     = mean cosine to k nearest seed papers      │
                            │ thresholds calibrated from the seed itself      │
                            │ (leave-one-out percentiles)                     │
                            └─────────────────────────────────────────────────┘
                                                     ▼
                       rank + cap per class ──▶ human review of flagged items
                                                     ▼
                    discovered.csv · candidates_scored.csv · BIBLIOGRAPHY.md
```

**Why two gates and calibration?** Naively searching OpenAlex returns thousands
of loosely-related AV papers. To stay *consistent with the survey's actual
scope* and avoid flooding the bibliography, a candidate must be (a) topically on
the edge-case theme and (b) at least as close to the existing corpus as a
*median paper the authors already cite*. Both thresholds are derived from the
seed corpus by leave-one-out, so the screening is principled and reproducible
rather than hand-picked. The curated set is then ranked and capped per class;
the full scored list is preserved in `candidates_scored.csv` for transparency.

In the most recent run: **2,836** records harvested → **2,580** after
de-duplication → **581** above the relevance floor → **32** curated additions
(top-15 per class), merged with the **263**-paper seed into a **295**-study
living bibliography.

---

## Taxonomy

| Class | Subsections |
|---|---|
| **Perception-related** | Reconstructive & Generative · Probability & Confidence Scores · Feature & Activation Extraction · Foundation Model & Adaptive Learning · Other |
| **Trajectory-related** | Surrogate Safety Metrics · Probability Estimation · Machine Learning · Challenging the System Under Test · Novel Scenario Generation |
| **Knowledge-driven** | Influencing Factors · Formalisation of Description · Qualification & Classification |

Anchor descriptions for each class/subsection are taken verbatim from the
survey (see [`src/classify.py`](src/classify.py)), so the data-driven assignment
*recovers the manual taxonomy* rather than inventing a new one.

---

## Repository layout

```
living-survey/
├── README.md
├── BIBLIOGRAPHY.md            # auto-generated, browsable, grouped by taxonomy
├── config.yaml               # search terms + tuning parameters (mirrors src defaults)
├── requirements.txt
├── src/
│   ├── harvest.py            # OpenAlex retrieval (Boolean phrase queries)
│   ├── classify.py          # shared taxonomy + Sentence-BERT classifier
│   ├── classify_llm.py      # OPTIONAL modular-LLM second opinion (needs API key)
│   ├── discover.py          # end-to-end: harvest → screen → classify → write
│   └── report.py            # build BIBLIOGRAPHY.md from the data
├── data/
│   ├── seed_corpus.csv       # 263 expert-curated papers (frozen)
│   ├── discovered.csv        # curated new papers (top-N per class)
│   ├── candidates_scored.csv # full scored candidate list (transparency)
│   └── bibliography.csv      # merged seed + discovered, classified
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
| `PER_CLASS_CAP` | curated picks per class | 15 |
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
