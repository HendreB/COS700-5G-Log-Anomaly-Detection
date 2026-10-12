"""
The benchmark: compare my TF-IDF prototype with keyword filtering.

Runs keyword filtering, my model fitted on the attack capture itself, and my model
trained on the normal capture first, on each attack capture (AMF, SMF, UPF) and on all three combined,
scores them against the ground truth, and writes:
    results/benchmark/benchmark_comparison.csv       - one row per dataset x method
    results/benchmark/benchmark_report.txt           - the formatted comparison table
    results/benchmark/benchmark_<dataset>_events.csv - every line with its label and each method's flag

With --sweep it instead runs the models at several cut-offs (SWEEP_PERCENTS) and writes:
    results/benchmark/benchmark_sweep.csv            - one row per dataset x method x cut-off
    results/benchmark/benchmark_sweep_report.txt     - the formatted sweep tables
    results/benchmark/benchmark_pr_curve.csv         - full precision-recall curve per model, for plotting
The normal benchmark files above are left untouched by a sweep.

Ground truth (--labels):
    timeline  - (default) every line labelled on its own, using the attack window in
                data/attack_timeline.log (see label_by_attack_window)
    templates - one label per template from data/ground_truth_labels.csv

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --outlier-percent 5 --repeats 10
    python scripts/run_benchmark.py --labels templates
    python scripts/run_benchmark.py --sweep
"""

from pathlib import Path
import argparse
import sys
import time

import pandas as pd

# Make sure the project root is on sys.path so the 'src' imports work when this is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (
    SWEEP_PERCENTS,
    build_benchmark_report,
    build_sweep_report,
    precision_recall_points,
    run_benchmark,
    scores_to_frame,
)
from src.evaluation.ground_truth import (
    ATTACK_TEMPLATE_SHARE,
    attach_labels,
    label_by_attack_window,
    load_attack_windows,
    load_labels,
)
from src.modeling.anomaly_detector import DEFAULT_N_CLUSTERS, DEFAULT_OUTLIER_PERCENT
from src.preprocessing.log_parser import list_available_logs, load_and_preprocess_logs, load_normal_logs

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark"  # all benchmark outputs in one subfolder
LABEL_PATH = DATA_DIR / "ground_truth_labels.csv"
TIMELINE_PATH = DATA_DIR / "attack_timeline.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TF-IDF prototype vs keyword filtering.")
    parser.add_argument("--clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help=f"K-Means micro-clusters (default: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--outlier-percent", type=float, default=DEFAULT_OUTLIER_PERCENT,
                        help=f"Share of events the model flags (default: {DEFAULT_OUTLIER_PERCENT})")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Timing repeats per method; the median is reported (default: 5)")
    parser.add_argument("--labels", choices=["timeline", "templates"], default="timeline",
                        help="Ground truth: per-line from attack_timeline.log (default) "
                             "or per-template from ground_truth_labels.csv")
    parser.add_argument("--sweep", action="store_true",
                        help=f"Run the models at cut-offs {SWEEP_PERCENTS} instead of one --outlier-percent")
    return parser.parse_args()


def run_sweep(prepared, args, ground_truth_note: str | None) -> None:
    """
    Run the benchmark once per cut-off in SWEEP_PERCENTS and save the combined table.

    Each cut-off goes through exactly the same run_benchmark code as a normal run, so the
    1% and 5% rows match benchmark_report.txt. Timing isn't the point here, so one repeat.
    """
    sweep_rows = []
    curves = []
    print(f"\nRunning the threshold sweep at {', '.join(f'{p:g}%' for p in SWEEP_PERCENTS)}...")
    for name, labelled, preprocess_seconds, baseline_df, baseline_seconds in prepared:
        for percent in SWEEP_PERCENTS:
            scores, events = run_benchmark(
                labelled,
                dataset=name,
                preprocess_seconds=preprocess_seconds,
                n_clusters=args.clusters,
                outlier_percent=percent,
                repeats=1,
                baseline_df=baseline_df,
                baseline_preprocess_seconds=baseline_seconds,
            )
            frame = scores_to_frame(scores)
            frame.insert(2, "outlier_percent", percent)
            sweep_rows.append(frame)
        # The distances don't depend on the cut-off, so the curve only needs the last run's events.
        curves.append(precision_recall_points(events, name))
        print(f"  - {name} done")

    sweep_df = pd.concat(sweep_rows, ignore_index=True)
    report = build_sweep_report(sweep_df, args.clusters, ground_truth_note)
    sweep_df.to_csv(RESULTS_DIR / "benchmark_sweep.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS_DIR / "benchmark_pr_curve.csv", index=False)
    (RESULTS_DIR / "benchmark_sweep_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nArtefacts persisted to '{RESULTS_DIR}':")
    print("  - benchmark_sweep.csv")
    print("  - benchmark_sweep_report.txt")
    print("  - benchmark_pr_curve.csv")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.labels == "timeline":
        windows = load_attack_windows(TIMELINE_PATH)
        window_text = "; ".join(f"{start:%Y-%m-%d %H:%M:%S} to {stop:%H:%M:%S}" for start, stop in windows)
        ground_truth_note = (
            f"per line, from {TIMELINE_PATH.name}: logged during the attack ({window_text}) "
            f"and template mostly (>= {ATTACK_TEMPLATE_SHARE:.0%}) seen during the attack"
        )
        label_lines = lambda df: label_by_attack_window(df, windows)
    else:
        labels = load_labels(LABEL_PATH)
        ground_truth_note = None
        label_lines = lambda df: attach_labels(df, labels)

    attack_files = [p for p in list_available_logs(DATA_DIR) if p.name.endswith("_attack_logs.txt")]
    if not attack_files:
        print(f"No *_attack_logs.txt files found in {DATA_DIR}")
        raise SystemExit(1)

    # Load each attack capture once and time the preprocessing, because the model
    # needs it and keyword search doesn't (that time gets added to the model's runtime).
    datasets: list[tuple[str, pd.DataFrame, float]] = []
    for path in attack_files:
        start = time.perf_counter()
        df = load_and_preprocess_logs(path)
        datasets.append((path.stem, df, time.perf_counter() - start))

    combined_df = pd.concat([df for _, df, _ in datasets], ignore_index=True)
    combined_seconds = sum(seconds for _, _, seconds in datasets)
    datasets.append(("combined_attack_logs", combined_df, combined_seconds))

    # The normal captures, per network function, for the "trained on normal" model.
    normal_logs: dict[str, tuple[pd.DataFrame, float]] = {}
    for nf in sorted(combined_df["network_function"].unique()):
        start = time.perf_counter()
        normal_logs[nf] = (load_normal_logs(DATA_DIR, {nf}), time.perf_counter() - start)

    # Label each dataset and pair it with its normal logs:
    # AMF attack logs get the AMF normal log, etc.; the combined set gets all of them.
    prepared = []
    for name, df, preprocess_seconds in datasets:
        nfs = sorted(df["network_function"].unique())
        baseline_df = pd.concat([normal_logs[nf][0] for nf in nfs], ignore_index=True)
        baseline_seconds = sum(normal_logs[nf][1] for nf in nfs)
        prepared.append((name, label_lines(df), preprocess_seconds, baseline_df, baseline_seconds))

    if args.sweep:
        run_sweep(prepared, args, ground_truth_note)
        return

    all_scores = []
    manual_lines = 0
    scored_lines = 0
    print("\nRunning keyword filtering and the TF-IDF model on each dataset...")
    for name, labelled, preprocess_seconds, baseline_df, baseline_seconds in prepared:
        scores, events = run_benchmark(
            labelled,
            dataset=name,
            preprocess_seconds=preprocess_seconds,
            n_clusters=args.clusters,
            outlier_percent=args.outlier_percent,
            repeats=args.repeats,
            baseline_df=baseline_df,
            baseline_preprocess_seconds=baseline_seconds,
        )
        all_scores.extend(scores)
        events.to_csv(RESULTS_DIR / f"benchmark_{name}_events.csv", index=False)

        # Only count the individual files for the "% manually labelled" figure,
        # otherwise the combined dataset would count every line twice.
        if name != "combined_attack_logs":
            manual_lines += int(labelled["is_manual"].sum())
            scored_lines += len(labelled)

    scores_df = scores_to_frame(all_scores)
    manual_share = manual_lines / scored_lines if scored_lines else 0.0
    report = build_benchmark_report(
        scores_df, manual_share, args.clusters, args.outlier_percent, args.repeats,
        ground_truth_note=ground_truth_note,
    )

    scores_df.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
    (RESULTS_DIR / "benchmark_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nArtefacts persisted to '{RESULTS_DIR}':")
    print("  - benchmark_comparison.csv")
    print("  - benchmark_report.txt")
    for name, _, _ in datasets:
        print(f"  - benchmark_{name}_events.csv")


if __name__ == "__main__":
    main()
