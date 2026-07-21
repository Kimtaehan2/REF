# When Repeated Exposure Is Not Fatigue

This directory contains the code used for the analyses in *When Repeated
Exposure Is Not Fatigue: Distinguishing Item Rejection from Session
Cessation*. It starts from the public KuaiRec 2.0 Big Matrix tables and
reproduces the repeated-exposure measures, behavioral association models,
session-cessation analyses, and threshold robustness checks.

## Contents

```text
.
├── run_all.py                 single entry point
├── requirements.txt          pinned Python dependencies
├── data/                      place the KuaiRec files here
├── src/
│   ├── prepare_features.py    outcomes, controls, and exposure measures
│   ├── association_models.py skip and non-skip watch-intensity models
│   ├── session_cessation.py   session boundary and cessation models
│   ├── within_between.py      within- and between-user decomposition
│   ├── threshold_analysis.py full-population threshold analysis
│   ├── estimators.py          fixed-effects threshold utilities
│   └── summarize_results.py   manuscript table assembly
├── reference_results/         submitted numerical results
└── results/                   generated output
```

The repository does not include the KuaiRec interaction files.

## Data setup

Download the KuaiRec 2.0 **Big Matrix** release and place the following files
in `data/`:

```text
data/
├── big_matrix.csv
├── item_categories.csv
└── item_daily_features.csv
```

The analysis uses these columns:

| File | Required columns |
|---|---|
| `big_matrix.csv` | `user_id`, `video_id`, `timestamp`, `date`, `watch_ratio`, `video_duration`, `play_duration` |
| `item_categories.csv` | `video_id`, `feat` |
| `item_daily_features.csv` | `video_id`, `date`, `play_cnt` |

The runner checks the filenames and headers before starting. The submitted
analysis used 7,176 users and 12,523,630 interactions after excluding the
first interaction of each user where an inter-event time was required.

## Environment

Python 3.11 is recommended. A clean virtual environment can be prepared with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The full analysis is memory intensive because the feature table contains more
than 12 million rows. At least 32 GB of RAM is recommended. The threshold
bootstrap is the longest step and may take several hours depending on the CPU
and BLAS implementation.

## Reproduce all results

From this directory, run one command:

```bash
python run_all.py --data-dir data
```

The command runs the following stages in order:

1. constructs the nine repeated-exposure families;
2. fits skip GEE and non-skip watch-intensity mixed models on the fixed
   1,000-user sample;
3. estimates the inactivity-gap mixture and cessation models;
4. fits the within- and between-user cessation decomposition;
5. estimates full-population thresholds, bootstrap intervals, and influence
   diagnostics;
6. writes the manuscript-ready summary.

If `results/features_timegated.parquet` already exists, feature construction is
skipped. Use `--force-features` to rebuild it. An existing feature file can be
supplied with `--features /path/to/features_timegated.parquet`.

## Output

The main files are:

| File | Description |
|---|---|
| `results/paper_results.md` | manuscript-ready representative estimates |
| `results/paper_table.csv` | the same estimates in machine-readable form |
| `results/table_associations.tex` | generated LaTeX for Table 2 |
| `results/association_results.csv` | all skip and watch-intensity models |
| `results/session_cessation_results.csv` | primary cessation models |
| `results/session_boundary_sensitivity.csv` | alternative session boundaries |
| `results/within_between_results.csv` | within- and between-user estimates |
| `results/threshold_results.csv` | thresholds, bootstrap intervals, and influence results |
| `results/run_manifest.json` | paths, seed, Python version, and timings |

To inspect the submitted numerical results without downloading KuaiRec, run:

```bash
python run_all.py --reference-only
```

This writes `paper_results.md`, `paper_table.csv`, and
`table_associations.tex` from the CSV files in `reference_results/`.

## Analysis conventions

- User sampling uses NumPy's random generator with seed 42.
- The association and cessation models use the same 1,000-user sample.
- Continuous exposure measures are standardized within the analysis sample.
- Skip is defined as `watch_ratio < 0.2`.
- Watch intensity is evaluated only for non-skipped interactions and is capped
  at 5.
- AUC uses five-fold cross-validation grouped by user.
- Session cessation is inferred from inactivity and is not an observed
  application-exit event.
- Threshold models use the full user population, a trimming fraction of 0.001,
  2,000 user cases-bootstrap draws, and 200 wild-bootstrap draws.

All reported estimates are observational associations rather than causal
effects.
