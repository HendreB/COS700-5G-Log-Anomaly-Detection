"""
Phase 4 — Head-to-head benchmark: TF-IDF prototype vs keyword filtering.

This answers research question 2.1.3. Both methods look at exactly the same log
lines and get scored against the same ground-truth labels (ground_truth.py), so the
comparison is like-for-like. I measure:
  - Accuracy, Precision, Recall, F1-score (detection quality)
  - Runtime in seconds (computational efficiency)

My model also gives every line a centroid distance, not only a yes/no flag, so I
report Average Precision for it too. That score doesn't depend on the 1% cut-off
I chose, which helps me explain whether a bad result comes from the model or the threshold.
"""

from __future__ import annotations

import contextlib
import io
import statistics
import time
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.evaluation.keyword_baseline import ALERT_KEYWORDS, keyword_filter
from src.modeling.anomaly_detector import train_and_detect


@dataclass
class MethodScores:
    """Metrics for one method on one dataset — one row of the comparison table."""

    dataset: str
    method: str
    events: int
    true_anomalies: int
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    runtime_seconds: float
    average_precision: float | None = None  # only the ML model produces a ranking


def score_method(dataset: str, method: str, y_true, y_pred, runtime: float) -> MethodScores:
    """Turn true labels and predicted flags into the standard classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # zero_division=0 so a method that flags nothing gets precision 0 instead of crashing.
    return MethodScores(
        dataset=dataset,
        method=method,
        events=len(y_true),
        true_anomalies=int(sum(y_true)),
        flagged=int(sum(y_pred)),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        runtime_seconds=runtime,
    )


def median_runtime(func, repeats: int) -> tuple[float, object]:
    """
    Run a function several times and return the median runtime plus the last result.

    A single timing on Windows can jump around a lot, so the median of a few runs
    is a fairer number to put in the report. train_and_detect prints progress
    messages, which I hide during the repeats so the terminal stays readable.
    """
    timings = []
    result = None
    for _ in range(max(1, repeats)):
        with contextlib.redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            result = func()
            timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


MODEL_SELF = "TF-IDF + K-Means"
MODEL_NORMAL = "TF-IDF + K-Means (trained on normal)"


def run_benchmark(
    labelled_df: pd.DataFrame,
    dataset: str,
    preprocess_seconds: float,
    n_clusters: int,
    outlier_percent: float,
    repeats: int = 5,
    baseline_df: pd.DataFrame | None = None,
    baseline_preprocess_seconds: float = 0.0,
) -> tuple[list[MethodScores], pd.DataFrame]:
    """
    Run all methods on the same labelled dataset and score them.

    Methods: keyword filtering, my model fitted on the attack capture itself, and
    (if baseline_df is given) my model trained on the normal capture first.

    Returns the score rows plus a per-line table (label, keyword flag, model flags)
    so I can go and look at exactly where the methods disagree.
    """
    y_true = labelled_df["final_label"].to_numpy()

    # --- Traditional keyword filtering (works straight on the raw lines) ---
    keyword_seconds, keyword = median_runtime(lambda: keyword_filter(labelled_df), repeats)

    # --- My TF-IDF + K-Means prototype ---
    # Its runtime includes preprocessing, because the model can't run without the
    # cleaned templates, whereas grep can. Leaving that out would be unfair to keyword search.
    model_seconds, detection = median_runtime(
        lambda: train_and_detect(
            labelled_df,
            n_clusters=n_clusters,
            outlier_percent=outlier_percent,
        ),
        repeats,
    )
    model_flags = detection.result_df["is_anomaly"].to_numpy()

    keyword_scores = score_method(
        dataset, "Keyword filtering", y_true, keyword.flags.to_numpy(), keyword_seconds
    )
    model_scores = score_method(
        dataset, MODEL_SELF, y_true, model_flags, model_seconds + preprocess_seconds
    )
    has_both_classes = 0 < y_true.sum() < len(y_true)
    if has_both_classes:
        model_scores.average_precision = average_precision_score(
            y_true, detection.result_df["centroid_distance"].to_numpy()
        )
    all_scores = [keyword_scores, model_scores]

    events = labelled_df[["network_function", "source_file", "raw_log", "log_template"]].copy()
    events["true_label"] = y_true
    events["label_is_manual"] = labelled_df["is_manual"].to_numpy()
    events["keyword_flag"] = keyword.flags.to_numpy()
    events["keyword_match"] = keyword.matched_keyword.to_numpy()
    events["model_flag"] = model_flags
    events["centroid_distance"] = detection.result_df["centroid_distance"].to_numpy()

    # --- My model again, but trained on the normal capture first (proposal section 4.3) ---
    # The timing includes training on the normal logs and preprocessing both sets, which is
    # the worst case. In practice the normal model would be trained once, ahead of time.
    if baseline_df is not None:
        normal_seconds, normal_detection = median_runtime(
            lambda: train_and_detect(
                labelled_df,
                n_clusters=n_clusters,
                outlier_percent=outlier_percent,
                baseline_df=baseline_df,
            ),
            repeats,
        )
        normal_flags = normal_detection.result_df["is_anomaly"].to_numpy()
        normal_scores = score_method(
            dataset, MODEL_NORMAL, y_true, normal_flags,
            normal_seconds + preprocess_seconds + baseline_preprocess_seconds,
        )
        if has_both_classes:
            normal_scores.average_precision = average_precision_score(
                y_true, normal_detection.result_df["centroid_distance"].to_numpy()
            )
        all_scores.append(normal_scores)
        events["normal_model_flag"] = normal_flags
        events["normal_model_distance"] = normal_detection.result_df["centroid_distance"].to_numpy()

    return all_scores, events


def scores_to_frame(all_scores: list[MethodScores]) -> pd.DataFrame:
    """Flatten the score objects into a table for the CSV export."""
    return pd.DataFrame([vars(score) for score in all_scores])


def build_benchmark_report(
    scores_df: pd.DataFrame,
    manual_share: float,
    n_clusters: int,
    outlier_percent: float,
    repeats: int,
    ground_truth_note: str | None = None,
) -> str:
    """
    Format the comparison as a plain-text report I can paste into my write-up.

    ground_truth_note describes rule-based per-line labels (from the attack timeline).
    Without it the labels came from the template CSV, so I show how many are manual.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("BENCHMARK: TF-IDF + K-MEANS PROTOTYPE vs TRADITIONAL KEYWORD FILTERING")
    lines.append("=" * 78)
    lines.append(f"Model settings     : k={n_clusters}, top {outlier_percent:g}% furthest events flagged")
    lines.append(f"Runtime            : median of {repeats} run(s); model time includes preprocessing")
    lines.append("                     (the 'trained on normal' time also includes training on the normal logs)")
    lines.append(f"Keyword rule       : {', '.join(ALERT_KEYWORDS)}")
    if ground_truth_note:
        lines.append(f"Ground truth       : {ground_truth_note}")
    else:
        lines.append(f"Manually labelled  : {manual_share:.1%} of scored log lines")
    if not ground_truth_note and manual_share < 1.0:
        lines.append("")
        lines.append("!! PROVISIONAL RESULTS — some labels are still the automatic suggestions.")
        lines.append("!! Review data/ground_truth_labels.csv before using these numbers in the report.")
    lines.append("")

    for dataset, group in scores_df.groupby("dataset", sort=False):
        first = group.iloc[0]
        lines.append(f"Dataset: {dataset}  ({first['events']:,} events, {first['true_anomalies']:,} labelled anomalous)")
        lines.append("-" * 78)
        lines.append(f"{'Method':38s} {'Flagged':>7s} {'TP':>4s} {'FP':>5s} {'FN':>4s} "
                     f"{'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Time(s)':>8s}")
        for _, row in group.iterrows():
            lines.append(
                f"{row['method']:38s} {row['flagged']:>7,} {row['true_positives']:>4,} "
                f"{row['false_positives']:>5,} {row['false_negatives']:>4,} "
                f"{row['accuracy']:>6.3f} {row['precision']:>6.3f} {row['recall']:>6.3f} "
                f"{row['f1']:>6.3f} {row['runtime_seconds']:>8.4f}"
            )
        for _, row in group[group["average_precision"].notna()].iterrows():
            lines.append(f"  Average Precision, {row['method']} (threshold-free): {row['average_precision']:.3f}")
        lines.append("")

    lines.append("How to read this:")
    lines.append("  - Precision: of the lines a method flagged, how many really were anomalous.")
    lines.append("  - Recall   : of the truly anomalous lines, how many the method found.")
    lines.append("  - Accuracy is inflated when anomalies are rare (flagging nothing still")
    lines.append("    scores high), so F1 is the fairer single number here.")
    lines.append("  - The model flags a fixed share of events, so its recall can't be higher than")
    lines.append("    (flagged / labelled anomalous) — check Average Precision to separate the")
    lines.append("    model's ranking quality from the choice of cut-off.")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
# The 1% cut-off is my choice, not something the data tells me. To show the result
# doesn't hinge on that one choice, I re-run the benchmark over a range of cut-offs and
# report all of them. I never pick the "best" cut-off with the labels — that would be
# tuning on the answer key, which isn't possible in a real incident.
# Up to 40% so the F1 curve visibly turns over (the attack is ~35% of the combined lines);
# stopping while F1 is still rising would hide where the method's limit is.
SWEEP_PERCENTS = [0.5, 1, 2, 5, 7, 10, 15, 20, 25, 30, 40]


def precision_recall_points(events: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Full precision-recall curve for each model on one dataset, for plotting.

    The centroid distance is the model's anomaly score, so sliding a threshold over it
    gives every possible precision/recall trade-off (the sweep table only shows a few).
    """
    y_true = events["true_label"].to_numpy()
    if not 0 < y_true.sum() < len(y_true):
        return pd.DataFrame()  # no curve without both normal and attack lines

    curves = []
    for method, score_column in [(MODEL_SELF, "centroid_distance"), (MODEL_NORMAL, "normal_model_distance")]:
        if score_column not in events:
            continue
        precision, recall, thresholds = precision_recall_curve(y_true, events[score_column].to_numpy())
        # precision_recall_curve returns one more P/R point than thresholds (the end point).
        curves.append(pd.DataFrame({
            "dataset": dataset,
            "method": method,
            "distance_threshold": list(thresholds) + [float("nan")],
            "precision": precision,
            "recall": recall,
        }))
    return pd.concat(curves, ignore_index=True)


def build_sweep_report(sweep_df: pd.DataFrame, n_clusters: int, ground_truth_note: str | None) -> str:
    """Format the sweep as one table per dataset: each model at every cut-off, keyword as reference."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("THRESHOLD SWEEP: TF-IDF + K-MEANS AT DIFFERENT CUT-OFFS vs KEYWORD FILTERING")
    lines.append("=" * 78)
    lines.append(f"Model settings     : k={n_clusters}, cut-offs {', '.join(f'{p:g}%' for p in SWEEP_PERCENTS)}")
    lines.append("Cut-off meaning    : fitted on the attack log -> flag the top N% furthest lines;")
    lines.append("                     trained on normal -> flag lines beyond the normal logs' (100-N)th percentile")
    if ground_truth_note:
        lines.append(f"Ground truth       : {ground_truth_note}")
    lines.append("Runtime is not reported here (one run per cut-off); see benchmark_report.txt for timings.")
    lines.append("")

    for dataset, group in sweep_df.groupby("dataset", sort=False):
        first = group.iloc[0]
        lines.append(f"Dataset: {dataset}  ({first['events']:,} events, {first['true_anomalies']:,} labelled anomalous)")
        lines.append("-" * 78)
        lines.append(f"{'Method':38s} {'Cut-off':>7s} {'Flagged':>7s} {'TP':>5s} {'FP':>5s} "
                     f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
        keyword = group[group["method"] == "Keyword filtering"].iloc[0]
        lines.append(
            f"{'Keyword filtering (no cut-off)':38s} {'-':>7s} {keyword['flagged']:>7,} "
            f"{keyword['true_positives']:>5,} {keyword['false_positives']:>5,} "
            f"{keyword['precision']:>6.3f} {keyword['recall']:>6.3f} {keyword['f1']:>6.3f}"
        )
        for method in [MODEL_SELF, MODEL_NORMAL]:
            rows = group[group["method"] == method].sort_values("outlier_percent")
            for _, row in rows.iterrows():
                lines.append(
                    f"{method:38s} {row['outlier_percent']:>6g}% {row['flagged']:>7,} "
                    f"{row['true_positives']:>5,} {row['false_positives']:>5,} "
                    f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}"
                )
        ap_rows = group[group["average_precision"].notna()].drop_duplicates("method")
        for _, row in ap_rows.iterrows():
            lines.append(f"  Average Precision, {row['method']} (same at every cut-off): {row['average_precision']:.3f}")
        lines.append("")

    lines.append("How to read this:")
    lines.append("  - A higher cut-off lets the model flag more lines: recall goes up, precision")
    lines.append("    usually goes down. Keyword filtering has no cut-off, so it has one fixed result.")
    lines.append("  - benchmark_pr_curve.csv holds the full precision-recall curve for plotting.")
    lines.append("=" * 78)
    return "\n".join(lines)
