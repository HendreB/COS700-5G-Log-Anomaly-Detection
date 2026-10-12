"""
SOC-style report generator for the prototype output.

Formats the detection results as an automated intelligence report — written in
third-person, professional tone, the way a commercial SOC tool would present findings.
The underlying data comes from anomaly_detector.py; this module only handles formatting.
"""


from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.modeling.anomaly_detector import DetectionResult

# OpenAir logs follow: [timestamp] [module] [level] message — this pulls out the level field.
LOG_LEVEL_PATTERN = re.compile(r"\]\s*\[(\w+)\s*\]\s*\[(\w+)\]", re.IGNORECASE)


def extract_log_level(raw_log: str) -> str:
    """Extract severity level from a raw log line."""
    match = LOG_LEVEL_PATTERN.search(raw_log)
    if match:
        return match.group(2).lower()
    return "unknown"


def describe_cluster(df_logs: pd.DataFrame, cluster_id: int) -> dict:
    """Summarise one micro-cluster baseline."""
    cluster_df = df_logs[df_logs["cluster"] == cluster_id]
    template_counts = (
        cluster_df["log_template"].value_counts().head(5).reset_index()
    )
    template_counts.columns = ["log_template", "count"]

    levels = cluster_df["raw_log"].apply(extract_log_level).value_counts()
    top_level = levels.index[0] if not levels.empty else "unknown"
    mean_distance = cluster_df["centroid_distance"].mean()

    return {
        "cluster_id": cluster_id,
        "size": len(cluster_df),
        "percentage": round(len(cluster_df) / len(df_logs) * 100, 2),
        "dominant_level": top_level,
        "mean_centroid_distance": round(float(mean_distance), 6),
        "top_templates": template_counts,
    }


def build_soc_report(
    dataset_name: str,
    detection: DetectionResult,
    source_files: list[str] | None = None,
) -> str:
    """Build an automated third-person SOC executive intelligence report."""
    df_logs = detection.result_df
    df_anomalies = detection.anomalies_df
    cluster_sizes = detection.cluster_sizes
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("AUTOMATED THREAT & ANOMALY INTELLIGENCE REPORT")
    lines.append("5G Core Network — Unsupervised Log Anomaly Detection")
    lines.append("=" * 78)
    # astimezone() attaches my PC's timezone so %z prints the real offset (e.g. +0200).
    lines.append(f"Report Generated    : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S (UTC%z)')}")
    lines.append(f"Dataset Identifier  : {dataset_name}")
    if source_files:
        lines.append(f"Source Artefacts    : {', '.join(source_files)}")
    lines.append(f"Total Events Parsed : {len(df_logs):,}")
    lines.append("Detection Method    : TF-IDF + K-Means Micro-Clustering + Centroid Distance")
    if detection.trained_on_baseline:
        lines.append(f"Normal Baseline     : model trained on {detection.baseline_events:,} normal-traffic events")
    lines.append("")

    # The wording for how anomalies were selected depends on which mode was used.
    if detection.trained_on_baseline:
        selection_summary = (
            f"lying further from the normal-traffic clusters than all but "
            f"{detection.outlier_percent:g}% of the normal logs"
        )
        selection_method = (
            f"  - Threshold: distance exceeded by only {detection.outlier_percent:g}% of normal-traffic events."
        )
    else:
        selection_summary = f"representing the furthest {detection.outlier_percent:g}% of events globally"
        selection_method = f"  - Global outlier selection: top {detection.outlier_percent:g}% furthest events flagged."

    lines.append("1. EXECUTIVE SUMMARY")
    lines.append("-" * 78)
    lines.append(
        f"The analyzed dataset comprises {len(df_logs):,} preprocessed log events. "
        f"The ML pipeline applied micro-clustering (k={detection.n_clusters_used}) to "
        "establish behavioural baselines, then scored each event by its Euclidean distance "
        "to the assigned cluster centroid in TF-IDF vector space."
    )
    lines.append("")
    lines.append(
        f"A total of {detection.outlier_count:,} structural anomalies were identified, "
        f"{selection_summary} "
        f"(distance threshold: {detection.distance_threshold:.6f}). "
        "These events exhibit the highest deviation from established micro-cluster "
        "centroids and warrant SOC review."
    )
    lines.append("")

    lines.append("2. DETECTION METHODOLOGY")
    lines.append("-" * 78)
    lines.append("Phase 2 — Preprocessing:")
    lines.append("  Regex-based template extraction normalizes dynamic fields (timestamps,")
    lines.append("  IMSI, IP addresses, file paths, hex codes) to isolate static log patterns.")
    lines.append("  Negation words (not, no, cannot) are kept so 'is not valid' differs from 'is valid'.")
    lines.append("")
    lines.append("Phase 3 — Feature Extraction & Unsupervised Clustering:")
    lines.append("  - TF-IDF vectorization (unigrams + bigrams) converts templates to feature vectors.")
    lines.append(f"  - K-Means micro-clustering (k={detection.n_clusters_used}) partitions normal behaviour.")
    lines.append("  - Per-event centroid distance computed in Euclidean space.")
    lines.append(selection_method)
    lines.append("")
    lines.append("Distance Statistics:")
    lines.append(f"  Mean centroid distance : {detection.mean_distance:.6f}")
    lines.append(f"  Maximum observed       : {detection.max_distance:.6f}")
    lines.append(f"  Anomaly threshold      : {detection.distance_threshold:.6f}")
    lines.append("")

    lines.append("3. MICRO-CLUSTER BASELINE ANALYSIS")
    lines.append("-" * 78)
    lines.append(
        "The following micro-clusters represent established behavioural baselines. "
        "Cluster identifiers are model-assigned and carry no inherent severity ranking."
    )
    lines.append("")

    for cluster_id in sorted(cluster_sizes.index):
        info = describe_cluster(df_logs, cluster_id)
        lines.append(
            f"Cluster {cluster_id:02d} | Events: {info['size']:,} ({info['percentage']}%) | "
            f"Dominant severity: {info['dominant_level']} | "
            f"Mean distance: {info['mean_centroid_distance']:.6f}"
        )
        lines.append("  Representative templates:")
        for _, row in info["top_templates"].head(2).iterrows():
            template = row["log_template"]
            if len(template) > 88:
                template = template[:85] + "..."
            lines.append(f"    [{row['count']:,}x] {template}")
        lines.append("")

    lines.append("4. CENTROID DISTANCE OUTLIER FINDINGS")
    lines.append("-" * 78)
    # Section 4 summarises the FULL anomaly set; Section 5 below only prints the
    # deduplicated, severity-sorted display list, so the counts there are smaller.
    df_display = detection.display_anomalies_df
    if not df_anomalies.empty:
        # Severity distribution among the flagged anomalies
        anomaly_levels = df_anomalies["raw_log"].apply(extract_log_level).value_counts()
        lines.append("Severity distribution among flagged structural anomalies:")
        for level, count in anomaly_levels.items():
            pct = round(count / len(df_anomalies) * 100, 1)
            lines.append(f"  {level.upper():8s}: {count:,} events ({pct}%)")

        # Proposed threat category summary (enrichment layer)
        if "proposed_threat_category" in df_anomalies.columns:
            lines.append("")
            lines.append("Proposed threat categories among flagged anomalies:")
            cat_counts = df_anomalies["proposed_threat_category"].value_counts()
            for cat, count in cat_counts.items():
                pct = round(count / len(df_anomalies) * 100, 1)
                # Count how many of this category are marked high risk
                high_risk_count = int(
                    df_anomalies.loc[df_anomalies["proposed_threat_category"] == cat, "high_risk"].sum()
                ) if "high_risk" in df_anomalies.columns else 0
                hr_note = f" ({high_risk_count} high-risk)" if high_risk_count else ""
                lines.append(f"  {cat}: {count:,} events ({pct}%)" + hr_note)

            # Overall high-risk summary
            if "high_risk" in df_anomalies.columns:
                total_high = int(df_anomalies["high_risk"].sum())
                lines.append("")
                lines.append(f"Total high-risk flagged anomalies: {total_high:,} of {len(df_anomalies):,}")

        if "network_function" in df_anomalies.columns:
            lines.append("")
            lines.append("Network function distribution:")
            nf_counts = df_anomalies["network_function"].value_counts()
            for nf, count in nf_counts.items():
                pct = round(count / len(df_anomalies) * 100, 1)
                lines.append(f"  {nf:8s}: {count:,} events ({pct}%)")

        if "source_file" in df_anomalies.columns:
            lines.append("")
            lines.append("Source artefact distribution:")
            src_counts = df_anomalies["source_file"].value_counts()
            for src, count in src_counts.items():
                lines.append(f"  {src}: {count:,} events")
    lines.append("")

    lines.append("5. STRUCTURAL ANOMALY EVENTS (SEVERITY-PRIORITIZED, UNIQUE TEMPLATES)")
    lines.append("-" * 78)
    lines.append(
        "Display policy: one representative event per normalized template. "
        "High-severity events ([fatal], [error], [warning], [reject]) are elevated "
        "above mathematical distance rank. Remaining slots ordered by centroid distance."
    )
    if df_display.empty:
        lines.append("No structural anomalies exceeded the detection threshold.")
    else:
        unique_templates = len(df_display)
        high_severity_count = int((df_display["severity_rank"] > 0).sum())
        lines.append(
            f"Unique anomaly templates displayed: {unique_templates} | "
            f"High-severity representatives: {high_severity_count}"
        )

        # Group displayed anomalies by the proposed threat category (includes high-risk tag)
        if "threat_display" in df_display.columns:
            grouped = df_display.groupby("threat_display", sort=False)
            grouped_iter = grouped
        else:
            grouped_iter = [("Uncategorized", df_display)]

        for threat_label, group in grouped_iter:
            lines.append("")
            lines.append(f"PROPOSED THREAT CATEGORY: {threat_label}")
            lines.append("-" * 78)

            for rank, (_, row) in enumerate(group.iterrows(), start=1):
                severity_label = "HIGH" if row["severity_rank"] > 0 else "STRUCTURAL"
                lines.append(
                    f"Anomaly {rank:02d} | Severity: {severity_label} | "
                    f"Distance: {row['centroid_distance']:.6f} | Cluster: {row['cluster']}"
                )
                lines.append(
                    f"  Template Occurrences : {int(row['template_occurrences'])} event(s) in anomaly set"
                )
                if "network_function" in row.index:
                    lines.append(f"  Network Function     : {row['network_function']}")
                if "source_file" in row.index:
                    lines.append(f"  Source Artefact      : {row['source_file']}")
                lines.append(f"  Raw Event            : {row['raw_log']}")
                lines.append(f"  Normalized Template  : {row['log_template']}")
                if row["severity_rank"] > 0:
                    lines.append(
                        "  Assessment           : High-severity structural deviation from "
                        f"micro-cluster {row['cluster']} baseline. Immediate SOC triage recommended."
                    )
                else:
                    lines.append(
                        "  Assessment           : Elevated centroid distance from "
                        f"micro-cluster {row['cluster']} baseline. Validate operational context."
                    )

    lines.append("")
    lines.append("6. RECOMMENDED SOC ACTIONS")
    lines.append("-" * 78)
    lines.append("  - Prioritize review of WARNING, ERROR, and FATAL severity events in Section 5.")
    lines.append("  - Correlate flagged events with network function (AMF/SMF/UPF) and timestamp.")
    lines.append("  - Validate whether high-distance events represent misconfigurations, protocol")
    lines.append("    violations, or expected operational variance before escalation.")
    lines.append("  - Cross-reference anomaly CSV artefacts for full event enumeration and audit trail.")
    lines.append("")

    lines.append("7. ARTEFACT REFERENCE")
    lines.append("-" * 78)
    stem = dataset_name.replace(".txt", "").replace(" ", "_")
    lines.append(f"  {stem}_anomalies.csv              - Flagged events with centroid distances")
    lines.append(f"  {stem}_cluster_distribution.csv  - Micro-cluster population counts")
    lines.append(f"  {stem}_summary.txt               - This intelligence report")
    lines.append(f"  {stem}_scored_events.csv         - Full scored dataset (optional export)")
    lines.append("")
    lines.append("=" * 78)
    lines.append("END OF REPORT — AUTOMATED GENERATION — REQUIRES HUMAN VALIDATION")
    lines.append("=" * 78)

    return "\n".join(lines)


def save_soc_report(
    output_path: Path,
    dataset_name: str,
    detection: DetectionResult,
    source_files: list[str] | None = None,
) -> str:
    """Write the SOC intelligence report to disk."""
    report = build_soc_report(dataset_name, detection, source_files)
    output_path.write_text(report, encoding="utf-8")
    return report


def print_soc_report(report: str) -> None:
    """Print the SOC intelligence report to the terminal."""
    print("\n")
    print(report)
