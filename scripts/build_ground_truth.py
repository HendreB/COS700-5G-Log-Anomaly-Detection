"""
Creates (or refreshes) the template-level label file for the benchmark.

Only needed for 'run_benchmark.py --labels templates'. My main results use the
per-line labels from data/attack_timeline.log, which need no label file.

Reads every *_attack_logs.txt file plus the normal captures, and writes
data/ground_truth_labels.csv with one row per unique template and a suggested label.
To label by hand, put 1 or 0 in the 'label' column.

Running it again is safe: labels I already typed in are kept.

Usage:
    python scripts/build_ground_truth.py
"""

from pathlib import Path
import sys

import pandas as pd

# Make sure the project root is on sys.path so the 'src' imports work when this is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ground_truth import build_label_table, merge_existing_labels
from src.preprocessing.log_parser import list_available_logs, load_and_preprocess_logs

DATA_DIR = PROJECT_ROOT / "data"
LABEL_PATH = DATA_DIR / "ground_truth_labels.csv"


def main() -> None:
    all_logs = list_available_logs(DATA_DIR)
    attack_files = [p for p in all_logs if p.name.endswith("_attack_logs.txt")]
    # Everything that isn't an attack capture counts as the normal (benign) reference.
    baseline_files = [p for p in all_logs if not p.name.endswith("_attack_logs.txt")]

    if not attack_files:
        print(f"No *_attack_logs.txt files found in {DATA_DIR}")
        raise SystemExit(1)

    attack_df = pd.concat([load_and_preprocess_logs(p) for p in attack_files], ignore_index=True)
    baseline_df = pd.concat([load_and_preprocess_logs(p) for p in baseline_files], ignore_index=True)

    table = build_label_table(attack_df, baseline_df)
    table = merge_existing_labels(table, LABEL_PATH)
    table.to_csv(LABEL_PATH, index=False, encoding="utf-8")

    manual = table["label"].astype(str).str.strip().isin(["0", "1"]).sum()
    suggested_positive = int((table["suggested_label"] == 1).sum())
    print(f"\nWrote {len(table):,} template rows to '{LABEL_PATH.relative_to(PROJECT_ROOT)}'.")
    print(f"  Manually labelled so far : {manual:,}")
    print(f"  Suggested anomalous      : {suggested_positive:,}")
    print("\nNext: open the CSV, put 1 (anomalous) or 0 (normal) in the 'label' column,")
    print("then run 'python scripts/run_benchmark.py --labels templates'.")


if __name__ == "__main__":
    main()
