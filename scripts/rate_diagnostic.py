"""
Rate diagnostic: is the registration storm a *volume* anomaly rather than a *wording* one?

My TF-IDF model judges each log line by its words. During the storm most AMF lines are
ordinary registration steps, worded exactly like a normal registration — there are just
far more of them. This script shows that directly, without changing the model:

  1. It counts how often each (network function, template) appears per time bucket in the
     attack capture (the "rate view").
  2. It compares each AMF template's rate during the attack window with its rate in the
     normal capture, next to the anomaly score my trained-on-normal model gives it
     (the "wording view").
  3. It draws one figure with both views on the same time axis.

Nothing here feeds back into the detector or the benchmark; it's purely an analysis.

Writes to results/rate_diagnostic/:
    template_rate_over_time.csv - count per (network_function, template, time bucket)
    amf_template_rate_vs_score.csv - per AMF template: normal rate, attack-window rate, ratio, model score
    rate_vs_wording.png         - the figure for the report

Usage:
    python scripts/rate_diagnostic.py
    python scripts/rate_diagnostic.py --bucket-seconds 1
"""

from pathlib import Path
import argparse
import contextlib
import io
import sys

import matplotlib

matplotlib.use("Agg")  # write the figure to a file; no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Make sure the project root is on sys.path so the 'src' imports work when this is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ground_truth import label_by_attack_window, load_attack_windows
from src.modeling.anomaly_detector import DEFAULT_N_CLUSTERS, train_and_detect
from src.preprocessing.log_parser import list_available_logs, load_and_preprocess_logs, load_normal_logs

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "rate_diagnostic"
TIMELINE_PATH = DATA_DIR / "attack_timeline.log"

# Three AMF templates I plot individually (matched by a piece of their text):
# an ordinary registration step, a line only the attack produces, and background traffic.
EXAMPLE_TEMPLATES = {
    "Ordinary registration step": "Received Registration Request message, handling",
    "Attack-only line (auth failure)": "Request Authentication Vectors failure",
    "Background (NRF heartbeat timer)": "Set a timer to the next Heart-beat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate view vs wording view of the attack capture.")
    parser.add_argument("--bucket-seconds", type=int, default=10,
                        help="Width of the time buckets in seconds (default: 10)")
    return parser.parse_args()


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """The parser replaces timestamps with <TIMESTAMP> in the template, so I read them back from raw_log."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["raw_log"].str.extract(r"^\[([^\]]+)\]")[0])
    return df


def quiet(func, *args, **kwargs):
    """Run a pipeline function without its progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bucket = f"{args.bucket_seconds}s"
    windows = load_attack_windows(TIMELINE_PATH)
    win_start, win_stop = windows[0]

    attack_files = [p for p in list_available_logs(DATA_DIR) if p.name.endswith("_attack_logs.txt")]
    attack = add_timestamps(pd.concat([quiet(load_and_preprocess_logs, p) for p in attack_files], ignore_index=True))
    attack = label_by_attack_window(attack, windows)

    # --- 1. Rate view: count per template per time bucket, for all three network functions ---
    attack["bucket_start"] = attack["timestamp"].dt.floor(bucket)
    rate = (
        attack.groupby(["network_function", "log_template", "bucket_start"]).size()
        .rename("count").reset_index()
    )
    rate.to_csv(RESULTS_DIR / "template_rate_over_time.csv", index=False)

    # --- 2. AMF: rate during the attack vs the normal capture, next to the model's score ---
    amf = attack[attack["network_function"] == "AMF"].reset_index(drop=True)
    normal_amf = add_timestamps(quiet(load_normal_logs, DATA_DIR, {"AMF"}))

    # Same trained-on-normal model as the benchmark, at the 1% and 5% cut-offs.
    detections = {
        percent: quiet(train_and_detect, amf, n_clusters=DEFAULT_N_CLUSTERS,
                       outlier_percent=percent, baseline_df=normal_amf)
        for percent in (1.0, 5.0)
    }
    amf["model_score"] = detections[1.0].result_df["centroid_distance"].to_numpy()

    normal_minutes = (normal_amf["timestamp"].max() - normal_amf["timestamp"].min()).total_seconds() / 60
    window_minutes = (win_stop - win_start).total_seconds() / 60
    in_window = amf["timestamp"].between(win_start, win_stop)

    per_template = amf.groupby("log_template").agg(
        attack_label=("final_label", "max"),
        model_score=("model_score", "mean"),
    )
    per_template["normal_rate_per_min"] = normal_amf.groupby("log_template").size() / normal_minutes
    per_template["window_rate_per_min"] = amf[in_window].groupby("log_template").size() / window_minutes
    per_template = per_template.fillna({"normal_rate_per_min": 0.0, "window_rate_per_min": 0.0})
    # Templates never seen in the normal capture get a ratio of "inf"; I keep them but flag it.
    per_template["rate_ratio"] = per_template["window_rate_per_min"] / per_template["normal_rate_per_min"]
    per_template["seen_in_normal"] = per_template["normal_rate_per_min"] > 0
    per_template = per_template.sort_values("window_rate_per_min", ascending=False)
    per_template.to_csv(RESULTS_DIR / "amf_template_rate_vs_score.csv")

    # --- 3. The figure: rate view (top two panels) vs wording view (bottom panel) ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.3]})
    for ax in axes:
        ax.axvspan(win_start, win_stop, color="tab:red", alpha=0.12, label="Attack window (timeline)")

    total = amf.set_index("timestamp").resample(bucket).size()
    axes[0].step(total.index, total.values, where="post", color="black")
    axes[0].set_ylabel(f"AMF lines per {args.bucket_seconds} s")
    axes[0].set_title("(a) Rate view: all AMF log lines over time")

    # The registration step and the auth failure happen once per fake UE, so their counts are
    # identical during the storm. A thick solid line under a dashed one keeps both visible.
    styles = [("tab:blue", "-", 4.0), ("tab:red", "--", 1.5), ("tab:green", "-", 1.5)]
    for (label, needle), (colour, linestyle, width) in zip(EXAMPLE_TEMPLATES.items(), styles):
        lines = amf[amf["log_template"].str.contains(needle, regex=False)]
        series = lines.set_index("timestamp").resample(bucket).size().reindex(total.index, fill_value=0)
        axes[1].step(series.index, series.values, where="post", color=colour, linestyle=linestyle,
                     linewidth=width, label=label)
    axes[1].set_ylabel(f"Lines per {args.bucket_seconds} s")
    axes[1].set_title("(b) Rate view: three individual templates")
    axes[1].legend(loc="upper left", fontsize=8)

    normal_lines = amf[amf["final_label"] == 0]
    attack_lines = amf[amf["final_label"] == 1]
    axes[2].scatter(normal_lines["timestamp"], normal_lines["model_score"], s=4, color="grey",
                    alpha=0.5, label="Normal line (ground truth)")
    axes[2].scatter(attack_lines["timestamp"], attack_lines["model_score"], s=4, color="tab:red",
                    alpha=0.5, label="Attack line (ground truth)")
    for percent, style in [(1.0, "--"), (5.0, ":")]:
        axes[2].axhline(detections[percent].distance_threshold, color="black", linestyle=style,
                        linewidth=1, label=f"Flag threshold at N = {percent:g}%")
    shares = "\n".join(
        f"N = {percent:g}%: {(attack_lines['model_score'] > det.distance_threshold).mean():.1%} of attack lines flagged"
        for percent, det in detections.items()
    )
    axes[2].text(0.99, 0.35, shares, transform=axes[2].transAxes, ha="right", va="center", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "grey"})
    axes[2].set_ylabel("Anomaly score\n(distance, trained on normal)")
    axes[2].set_title("(c) Wording view: the TF-IDF model's score for every AMF line")
    axes[2].legend(loc="lower left", fontsize=8, markerscale=3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[2].set_xlabel(f"Time on {win_start:%Y-%m-%d} (attack capture)")
    fig.suptitle("Registration storm: obvious in the rate view, mostly hidden in the wording view", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rate_vs_wording.png", dpi=200)

    # --- Terminal summary ---
    attack_templates = per_template[per_template["attack_label"] == 1]
    seen = attack_templates[attack_templates["seen_in_normal"]]
    above_5 = (attack_lines["model_score"] > detections[5.0].distance_threshold).mean()
    peak = total.max()
    outside = total[(total.index < win_start) | (total.index > win_stop)]
    print("Rate diagnostic (AMF attack capture)")
    print(f"  Busiest {args.bucket_seconds}s bucket: {peak:,} lines; median outside the attack window: "
          f"{outside.median():.0f} lines")
    print(f"  Attack templates: {len(attack_templates)}, of which {len(seen)} also occur in the normal capture")
    print(f"  Median rate increase of those {len(seen)} during the attack: "
          f"{seen['rate_ratio'].median():.0f}x their normal-capture rate")
    print(f"  Attack lines scoring above the N=5% threshold (wording view): {above_5:.1%}")
    print(f"\nArtefacts persisted to '{RESULTS_DIR}':")
    print("  - template_rate_over_time.csv")
    print("  - amf_template_rate_vs_score.csv")
    print("  - rate_vs_wording.png")


if __name__ == "__main__":
    main()
