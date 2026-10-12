"""
Runs my detection pipeline on the three attack captures combined.

main.py's '--log all' would also pull in the normal captures, so this script picks
up only the *_attack_logs.txt files (AMF + SMF + UPF recorded during the attacks)
and produces the same four artefacts as main.py, under the name "combined_attack_logs".

Usage:
    python scripts/run_combined_attack_logs.py
"""

from pathlib import Path
import sys
import pandas as pd

# Make sure the project root is on sys.path so the 'src' imports work when this is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.log_parser import list_available_logs, load_and_preprocess_logs
from src.modeling.anomaly_detector import DEFAULT_N_CLUSTERS, DEFAULT_OUTLIER_PERCENT, train_and_detect
from src.reporting.summary_report import save_soc_report, print_soc_report

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Only the attack captures — the normal logs don't end in _attack_logs.txt.
log_files = [p for p in list_available_logs(DATA_DIR) if p.name.endswith("_attack_logs.txt")]
if not log_files:
    print(f"No attack log files found in {DATA_DIR}")
    raise SystemExit(1)

frames = [load_and_preprocess_logs(p) for p in log_files]
combined = pd.concat(frames, ignore_index=True)

print(f"Combined attack dataset: {len(combined):,} events from {len(log_files)} files")

# Same model settings as main.py's defaults (k=15, top 1%), so the results are comparable.
detection = train_and_detect(combined, n_clusters=DEFAULT_N_CLUSTERS, outlier_percent=DEFAULT_OUTLIER_PERCENT)

# --- Save the same artefacts main.py produces, in their own results/<name>/ folder ---
stem = "combined_attack_logs"
run_dir = RESULTS_DIR / stem
run_dir.mkdir(parents=True, exist_ok=True)

anomalies_path = run_dir / f"{stem}_anomalies.csv"
clusters_path = run_dir / f"{stem}_cluster_distribution.csv"
scored_path = run_dir / f"{stem}_scored_events.csv"

anomalies_path.write_text(detection.anomalies_df.to_csv(index=False), encoding='utf-8')
clusters_path.write_text(detection.cluster_sizes.to_csv(header=["count"]), encoding='utf-8')
scored_path.write_text(detection.result_df.to_csv(index=False), encoding='utf-8')

# Write the SOC report to disk and print it.
report = save_soc_report(run_dir / f"{stem}_summary.txt", stem, detection, source_files=[p.name for p in log_files])
print_soc_report(report)

print(f"\nArtefacts persisted to results/{stem}/:")
print(f" - {anomalies_path.name}")
print(f" - {clusters_path.name}")
print(f" - {scored_path.name}")
print(f" - {stem}_summary.txt")
