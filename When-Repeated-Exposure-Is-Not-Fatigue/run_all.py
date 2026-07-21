"""Run the complete analysis pipeline from the KuaiRec source tables."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR / "src"
DEFAULT_RESULTS_DIR = PACKAGE_DIR / "results"

REQUIRED_FILES = {
    "big_matrix.csv": {
        "user_id", "video_id", "timestamp", "date", "watch_ratio",
        "video_duration", "play_duration",
    },
    "item_categories.csv": {"video_id", "feat"},
    "item_daily_features.csv": {"video_id", "date", "play_cnt"},
}


def validate_data(data_dir):
    problems = []
    for filename, required_columns in REQUIRED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = set(next(csv.reader(handle)))
        missing_columns = sorted(required_columns - header)
        if missing_columns:
            problems.append(f"{filename} is missing columns: {', '.join(missing_columns)}")
    if problems:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(problems))


def run_step(label, script, env, extra_args=None):
    command = [sys.executable, str(SOURCE_DIR / script)] + list(extra_args or [])
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    started = time.time()
    subprocess.run(command, check=True, env=env, cwd=PACKAGE_DIR)
    elapsed = time.time() - started
    print(f"[{label}] completed in {elapsed:.1f} seconds", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the analyses for 'When Repeated Exposure Is Not Fatigue'."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_DIR / "data",
        help="Directory containing the three KuaiRec CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generated features and results.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Optional existing features_timegated.parquet file.",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Rebuild the feature Parquet even if it already exists.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Build the paper summary from the included reference CSV files without raw data.",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_only:
        run_step(
            "summary",
            "summarize_results.py",
            os.environ.copy(),
            [
                "--source-dir", str(PACKAGE_DIR / "reference_results"),
                "--output-dir", str(args.results_dir),
            ],
        )
        return

    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    validate_data(data_dir)

    feature_path = (
        args.features.expanduser().resolve()
        if args.features
        else results_dir / "features_timegated.parquet"
    )
    if args.features and not feature_path.exists():
        raise SystemExit(f"Feature file does not exist: {feature_path}")

    env = os.environ.copy()
    env["KUAIREC_DATA_DIR"] = str(data_dir)
    env["REVIEW_RESULTS_DIR"] = str(results_dir)
    env["KUAIREC_FEATURES"] = str(feature_path)

    timings = {}
    if args.force_features or not feature_path.exists():
        timings["prepare_features"] = run_step("features", "prepare_features.py", env)
    else:
        print(f"Using existing feature file: {feature_path}")

    timings["associations"] = run_step("associations", "association_models.py", env)
    timings["session_cessation"] = run_step("cessation", "session_cessation.py", env)
    timings["within_between"] = run_step("within-between", "within_between.py", env)
    timings["thresholds"] = run_step("thresholds", "threshold_analysis.py", env)
    timings["summary"] = run_step(
        "summary",
        "summarize_results.py",
        env,
        ["--source-dir", str(results_dir), "--output-dir", str(results_dir)],
    )

    manifest = {
        "python": sys.version,
        "data_dir": str(data_dir),
        "feature_file": str(feature_path),
        "random_seed": 42,
        "analysis_users": 1000,
        "timings_seconds": timings,
    }
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nAll analyses completed. Main summary: {results_dir / 'paper_results.md'}")


if __name__ == "__main__":
    main()
