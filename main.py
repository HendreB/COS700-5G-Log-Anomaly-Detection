"""
COS700 Research Prototype — Main entry point.

Ties together preprocessing, ML detection, and SOC report generation.
Run this to demo the full pipeline to my supervisor.
The keyword-filtering comparison lives separately in scripts/run_benchmark.py.

Usage:
    python main.py                              # AMF logs, k=15, top 1% outliers
    python main.py --log all                    # Combined AMF + SMF + UPF dataset
    python main.py --log smf_normal_logs.txt
    python main.py --log all --auto-clusters    # Silhouette-optimized k
    python main.py --clusters 20 --outlier-percent 0.5
    python main.py --log smf_attack_logs.txt --train-on-normal   # learn normal from smf_normal_logs.txt first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.integrity import check_unchanged, hash_files
from src.modeling.anomaly_detector import DEFAULT_N_CLUSTERS, DEFAULT_OUTLIER_PERCENT, train_and_detect
from src.preprocessing.log_parser import (
    list_available_logs,
    load_all_logs,
    load_and_preprocess_logs,
    load_normal_logs,
)
from src.reporting.summary_report import print_soc_report, save_soc_report

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"       # Real OpenAirInterface 5G core logs live here
RESULTS_DIR = PROJECT_ROOT / "results" # CSV reports and SOC summary written here

DEFAULT_LOG = "amf_normal_logs.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5G core log anomaly detection — TF-IDF micro-clustering + centroid outliers."
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG,
        help="Log file in data/, or 'all' to analyse every log file combined (default: AMF)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"K-Means micro-clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    parser.add_argument(
        "--outlier-percent",
        type=float,
        default=DEFAULT_OUTLIER_PERCENT,
        help=f"Global outlier selection — furthest N%% by centroid distance (default: {DEFAULT_OUTLIER_PERCENT})",
    )
    parser.add_argument(
        "--auto-clusters",
        action="store_true",
        help="Estimate optimal k via silhouette score before clustering",
    )
    parser.add_argument(
        "--train-on-normal",
        action="store_true",
        help="Learn normal behaviour from the clean logs of the same NF(s), then score --log against it",
    )
    return parser.parse_args()


def print_technical_report(detection, dataset_name: str) -> None:
    """Print concise technical metrics to the terminal."""
    cluster_distribution = pd.DataFrame(
        {
            "cluster": detection.cluster_sizes.index,
            "count": detection.cluster_sizes.values,
            "percentage": (
                detection.cluster_sizes.values / len(detection.result_df) * 100
            ).round(2),
        }
    )

    print("\n--- Technical Detection Metrics ---")
    print(f"Dataset              : {dataset_name}")
    print(f"Events analysed      : {len(detection.result_df):,}")
    print(f"Micro-clusters (k)   : {detection.n_clusters_used}")
    # Show the share actually flagged — with ties or --train-on-normal it isn't exactly --outlier-percent.
    flagged_share = detection.outlier_count / len(detection.result_df) * 100
    print(f"Outliers flagged     : {detection.outlier_count:,} ({flagged_share:.2f}%)")
    print(f"Distance threshold   : {detection.distance_threshold:.6f}")

    print("\nMicro-cluster distribution (top 10 by size):")
    print(cluster_distribution.sort_values("count", ascending=False).head(10).to_string(index=False))

    if not detection.display_anomalies_df.empty:
        print("\nPriority structural anomalies (severity-first, unique templates):")
        cols = ["severity_rank", "centroid_distance", "cluster", "raw_log"]
        if "network_function" in detection.display_anomalies_df.columns:
            cols.insert(3, "network_function")
        print(detection.display_anomalies_df[cols].head(5).to_string(index=False))


def resolve_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Load one log file or merge the logs depending on --log argument.

    Note: 'all' loads EVERY .txt in data/, which now includes the *_attack_logs.txt
    captures as well as the three normal ones.
    """
    if args.log.lower() == "all":
        df_logs = load_all_logs(DATA_DIR)
        source_files = sorted(df_logs["source_file"].unique().tolist())
        return df_logs, "combined_5g_core", source_files

    log_path = DATA_DIR / args.log
    if not log_path.is_file():
        available = list_available_logs(DATA_DIR)
        print(f"Error: Log file not found at '{log_path}'")
        if available:
            print("\nAvailable logs in data/:")
            for path in available:
                print(f"  - {path.name}")
            print("\nTip: use --log all to analyse every log file together.")
        else:
            print("\nPlace .txt log files in the 'data/' folder and retry.")
        raise SystemExit(1)

    df_logs = load_and_preprocess_logs(log_path)
    return df_logs, log_path.stem, [log_path.name]


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Initiating 5G log anomaly detection pipeline...")

    # Fingerprint every log in data/ before reading anything, so I can show afterwards
    # that the analysis didn't change the evidence.
    evidence_paths = list_available_logs(DATA_DIR)
    hashes_before = hash_files(evidence_paths)

    # --- Phase 2: Load and preprocess raw logs into templates ---
    df_logs, output_stem, source_files = resolve_dataset(args)

    # Optional: learn "normal" from the clean captures of the same network functions.
    # The file(s) being analysed are left out so the model never trains on what it scores.
    baseline_df = None
    if args.train_on_normal:
        baseline_df = load_normal_logs(DATA_DIR, set(df_logs["network_function"]))
        baseline_df = baseline_df[~baseline_df["source_file"].isin(source_files)]
        if baseline_df.empty:
            print("Error: --train-on-normal needs normal logs other than the one being analysed.")
            print("Use it on an attack capture, e.g. --log smf_attack_logs.txt --train-on-normal")
            raise SystemExit(1)
        print(f"Training baseline: {len(baseline_df):,} normal events from "
              f"{', '.join(sorted(baseline_df['source_file'].unique()))}")
        # Separate file names so these results don't overwrite the default-mode ones.
        output_stem = f"{output_stem}_vs_normal"

    # --- Phase 3: TF-IDF + micro-clustering + centroid outlier detection ---
    detection = train_and_detect(
        df_logs,
        n_clusters=args.clusters,
        outlier_percent=args.outlier_percent,
        auto_clusters=args.auto_clusters,
        baseline_df=baseline_df,
    )

    print_technical_report(detection, output_stem)

    # --- Export artefacts for analysis and the supervisor demo ---
    # Each run gets its own folder (results/<dataset>/) so results/ doesn't turn into
    # one long list of 50+ files. Re-running the same dataset overwrites its folder.
    run_dir = RESULTS_DIR / output_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    anomalies_path = run_dir / f"{output_stem}_anomalies.csv"
    clusters_path = run_dir / f"{output_stem}_cluster_distribution.csv"
    scored_path = run_dir / f"{output_stem}_scored_events.csv"
    summary_path = run_dir / f"{output_stem}_summary.txt"

    detection.anomalies_df.to_csv(anomalies_path, index=False)
    detection.cluster_sizes.to_csv(clusters_path, header=["count"])
    detection.result_df.to_csv(scored_path, index=False)

    integrity_ok = check_unchanged(evidence_paths, hashes_before)
    # List the fingerprint of every file this run actually used, including the normal baseline.
    used_files = list(source_files)
    if baseline_df is not None:
        used_files += sorted(baseline_df["source_file"].unique())
    used_hashes = {name: hashes_before[name] for name in used_files}

    # --- Phase 4: Generate and print the SOC intelligence report ---
    report = save_soc_report(
        summary_path,
        output_stem,
        detection,
        source_files=source_files,
        source_hashes=used_hashes,
        integrity_verified=integrity_ok,
    )
    print_soc_report(report)

    print(f"\nArtefacts persisted to '{run_dir}':")
    print(f"  - {anomalies_path.name}")
    print(f"  - {clusters_path.name}")
    print(f"  - {scored_path.name}")
    print(f"  - {summary_path.name}")


if __name__ == "__main__":
    main()
