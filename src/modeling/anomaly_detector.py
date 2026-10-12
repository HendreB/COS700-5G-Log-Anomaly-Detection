"""
Phase 3 — Feature extraction and unsupervised anomaly detection.

This is the core ML logic for my research prototype:
  1. Convert log templates to TF-IDF vectors (NLP feature extraction)
  2. Micro-cluster with K-Means (k=15) to model normal 5G behaviour
  3. Score each log by how far it sits from its cluster centroid
  4. Flag the furthest events (1% by default) as structural anomalies

I moved away from "smallest cluster = anomaly" because that was catching entire
blocks of benign startup logs. Centroid distance on individual events works better.

Optionally the model can learn "normal" from the clean captures first and then score
the attack captures against it (baseline_df in train_and_detect).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score

DEFAULT_N_CLUSTERS = 15       # Micro-clusters — tighter than k=3, learned from testing
DEFAULT_OUTLIER_PERCENT = 1.0 # Flag the furthest 1% of logs globally
DEFAULT_DISPLAY_COUNT = 10    # How many unique anomalies to show in the SOC report
SILHOUETTE_SAMPLE_SIZE = 5000 # Cap sample size when auto-selecting k (performance)

# Traditional severity flags — used to re-rank the report display, not to change
# which logs are flagged. The top 1% is still selected purely by centroid distance.
HIGH_SEVERITY_PATTERN = re.compile(
    r"\[(fatal|error|warning|reject)\]",
    re.IGNORECASE,
)
SEVERITY_RANK = {
    "fatal": 4,
    "error": 3,
    "reject": 3,
    "warning": 2,
}

# sklearn's built-in English stop list also removes negations like "not" and "no".
# In logs those words carry the whole meaning ("is valid" vs "is not valid",
# "available" vs "not available"), so I keep them in.
NEGATION_WORDS = {"not", "no", "nor", "cannot", "never", "without"}
LOG_STOP_WORDS = sorted(ENGLISH_STOP_WORDS - NEGATION_WORDS)

# Strips "[timestamp] [module] [level] " so the threat signatures only see the message.
LOG_PREFIX_PATTERN = re.compile(r"^(?:\[[^\]]*\]\s*){3}")

# Keyword signatures for the proposed threat category (first match wins).
# \b word boundaries stop "syn" matching "asynch" or "ping" matching "stopping".
THREAT_SIGNATURES = [
    # Signs of flooding traffic (e.g. hping3 UDP floods against the core).
    ("Volumetric DDoS / UDPFlood", re.compile(r"\b(?:udp|flood\w*|hping\w*)\b", re.IGNORECASE)),
    # Signs of SYN or ICMP (ping) flooding.
    ("Volumetric DDoS / SYN & ICMP Flood", re.compile(r"\b(?:syn|icmp|ping)\b", re.IGNORECASE)),
    # Signs of someone probing ports / refused connections. I dropped plain "tcp" —
    # it only ever matched normal NF profile JSON ("transport":"TCP").
    ("Port Scanning / DoS", re.compile(r"\b(?:scan\w*|port scan|connection refused)\b", re.IGNORECASE)),
    # Socket and config failures that show a network function struggling or breaking down.
    ("Resource Exhaustion / Core Breakdown", re.compile(
        r"invalid prefix|prefix length|socket bind|recvfrom failed|shutdown on socket"
        r"|transport endpoint is not connected|operation not permitted",
        re.IGNORECASE,
    )),
    # Rejected or failed procedures (registration, session setup, etc.).
    ("Authentication / Access Violation", re.compile(r"\b(?:unauthori[sz]ed|reject\w*|fail\w*)\b", re.IGNORECASE)),
]
HIGH_RISK_CATEGORIES = {
    "Volumetric DDoS / UDPFlood",
    "Volumetric DDoS / SYN & ICMP Flood",
    "Resource Exhaustion / Core Breakdown",
}


@dataclass
class DetectionResult:
    """Everything the pipeline returns — passed to the report generator and CSV export."""

    anomalies_df: pd.DataFrame          # Every flagged event
    cluster_sizes: pd.Series            # How many logs landed in each micro-cluster
    result_df: pd.DataFrame             # Every log with cluster + distance scores
    n_clusters_used: int
    outlier_count: int
    outlier_percent: float
    distance_threshold: float           # Minimum distance to qualify as an anomaly
    max_distance: float
    mean_distance: float
    display_anomalies_df: pd.DataFrame  # Deduped, severity-sorted list for the report
    trained_on_baseline: bool = False   # True when "normal" was learned from clean logs
    baseline_events: int = 0            # How many normal logs the model was trained on


def severity_priority(text: str) -> int:
    """
    Assign a numeric severity rank based on traditional high-severity log flags.

    Checks both raw events and normalized templates for [fatal], [error],
    [warning], and [reject] tokens.
    """
    matches = HIGH_SEVERITY_PATTERN.findall(text)
    if not matches:
        return 0
    return max(SEVERITY_RANK.get(match.lower(), 1) for match in matches)


def prepare_display_anomalies(
    anomalies_df: pd.DataFrame,
    max_display: int = DEFAULT_DISPLAY_COUNT,
) -> pd.DataFrame:
    """
    Build the list that actually gets printed in Section 5 of the SOC report.

    TF-IDF tends to rank one-off startup/config templates as the furthest outliers,
    so this function deduplicates by template and bumps [error]/[warning] events
    to the top so the report shows a useful variety of threats.
    """
    if anomalies_df.empty:
        return anomalies_df.copy()

    template_counts = anomalies_df["log_template"].value_counts()

    ranked = anomalies_df.copy()
    ranked["_severity_rank"] = ranked.apply(
        lambda row: max(
            severity_priority(str(row["raw_log"])),
            severity_priority(str(row["log_template"])),
        ),
        axis=1,
    )

    ranked = ranked.sort_values(
        ["_severity_rank", "centroid_distance"],
        ascending=[False, False],
    )
    # Keep one row per unique template — the one with highest severity/distance.
    unique = ranked.drop_duplicates(subset=["log_template"], keep="first")

    display = (
        unique.sort_values(
            ["_severity_rank", "centroid_distance"],
            ascending=[False, False],
        )
        .head(max_display)
        .copy()
    )
    display["template_occurrences"] = display["log_template"].map(template_counts)
    display["severity_rank"] = display["_severity_rank"]
    return display.drop(columns=["_severity_rank"])


def estimate_optimal_clusters(
    tfidf_matrix,
    min_k: int = 5,
    max_k: int = 20,
    random_state: int = 42,
) -> int:
    """
    Estimate an optimal micro-cluster count using silhouette score on a sample.

    Falls back to DEFAULT_N_CLUSTERS when the sample is too small to evaluate.
    """
    n_samples = tfidf_matrix.shape[0]
    upper_k = min(max_k, n_samples - 1)
    lower_k = min(min_k, upper_k)

    if upper_k < 2:
        return max(2, min(DEFAULT_N_CLUSTERS, n_samples))

    if n_samples > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n_samples, SILHOUETTE_SAMPLE_SIZE, replace=False)
        sample_matrix = tfidf_matrix[sample_idx]
    else:
        sample_matrix = tfidf_matrix

    best_k = DEFAULT_N_CLUSTERS
    best_score = -1.0

    for k in range(lower_k, upper_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(sample_matrix)
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            sample_matrix,
            labels,
            metric="euclidean",
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, sample_matrix.shape[0]),
        )
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def compute_centroid_distances(
    tfidf_matrix,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
) -> np.ndarray:
    """
    Measure how far each log sits from the centre of its assigned micro-cluster.

    A high distance means the log's wording doesn't fit the normal pattern for
    that cluster — even if the cluster itself is large and mostly benign.
    """
    n_samples = tfidf_matrix.shape[0]
    distances = np.empty(n_samples, dtype=np.float64)

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        cluster_vectors = tfidf_matrix[indices]
        center = cluster_centers[cluster_id].reshape(1, -1)
        cluster_distances = pairwise_distances(
            cluster_vectors, center, metric="euclidean"
        ).ravel()
        distances[indices] = cluster_distances

    return distances


def build_vectorizer(n_samples: int) -> TfidfVectorizer:
    """Set up the TF-IDF vectorizer the same way for both detection modes."""
    # min_df=2 on real data drops words that only appear in a single log line (usually
    # leftover IDs the regex missed). On tiny test inputs I keep everything (min_df=1),
    # otherwise the vocabulary can end up empty.
    min_df = 1 if n_samples < 50 else 2

    # TF-IDF turns each log template into a numeric vector the clusterer can use.
    # ngram_range=(1,2) captures both single words and word pairs like "registration request".
    # max_df=0.95 drops words that appear in almost every line (like "debug") since
    # they can't tell events apart anyway.
    return TfidfVectorizer(
        stop_words=LOG_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
    )


def classify_threat(raw_log: str) -> str:
    """Give a flagged log a *proposed* 5G threat category based on the words in it.

    This only adds a label for the analyst to read. It runs after the model has
    already decided what's anomalous, so it can't change which logs get flagged.
    Checks go top to bottom and the first match wins, so the order is the priority.
    """
    # Only look at the message text. The first version searched the whole line, so the
    # OAI module name "[udp    ]" alone made normal socket logs "Volumetric DDoS".
    message = LOG_PREFIX_PATTERN.sub("", str(raw_log))
    for category, pattern in THREAT_SIGNATURES:
        if pattern.search(message):
            return category
    # Mathematically unusual but none of my signatures match.
    return "Unknown Structural Anomaly"


def train_and_detect(
    log_df: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    outlier_percent: float = DEFAULT_OUTLIER_PERCENT,
    auto_clusters: bool = False,
    baseline_df: pd.DataFrame | None = None,
) -> DetectionResult:
    """
    Execute micro-clustering and centroid-distance outlier detection.

    Two modes:
      - Default (baseline_df=None): learn the clusters from log_df itself and flag
        the furthest outlier_percent of it. This is what the first prototype did.
      - Trained on normal logs (baseline_df given): learn TF-IDF + clusters ONLY from
        the clean capture, then score log_df against that model. The threshold is set
        so that outlier_percent of the NORMAL logs would be flagged, and any log_df
        event further out than that counts as an anomaly. This matches section 4.3 of
        my proposal ("define clusters that represent normal behaviour"). It also means
        the number of flags isn't fixed: a clean log gets few, an unusual one gets many.

    Args:
        log_df: DataFrame with 'log_template' and 'raw_log' columns.
        n_clusters: Target number of K-Means micro-clusters.
        outlier_percent: Percentage of furthest logs flagged (default 1%).
        auto_clusters: If True, estimate k via silhouette score before clustering.
        baseline_df: Optional normal-traffic logs to learn "normal" from.

    Returns:
        DetectionResult containing flagged anomalies and full scored dataset.
    """
    if log_df.empty:
        raise ValueError("No log entries to analyse.")

    trained_on_baseline = baseline_df is not None
    train_df = baseline_df if trained_on_baseline else log_df
    if train_df.empty:
        raise ValueError("Baseline (normal) log set is empty.")

    n_train = len(train_df)
    n_samples = len(log_df)

    vectorizer = build_vectorizer(n_train)
    train_matrix = vectorizer.fit_transform(train_df["log_template"])
    print(f"TF-IDF feature matrix: {train_matrix.shape[0]:,} samples x {train_matrix.shape[1]:,} features")

    if auto_clusters:
        n_clusters = estimate_optimal_clusters(train_matrix)
        print(f"Auto-selected micro-clusters (silhouette): k={n_clusters}")

    # Don't allow more clusters than the data can realistically support —
    # roughly at least 10 logs per cluster, and never fewer than 2 clusters.
    effective_clusters = min(n_clusters, max(2, n_train // 10), n_train)
    effective_clusters = max(2, effective_clusters)

    # K-Means groups similar log templates — each cluster becomes a "normal behaviour" baseline.
    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=42,
        n_init="auto",
    )
    train_labels = kmeans.fit_predict(train_matrix)
    train_distances = compute_centroid_distances(
        train_matrix, train_labels, kmeans.cluster_centers_
    )

    if trained_on_baseline:
        # Score the logs under investigation against the clusters learned from normal traffic.
        # The vocabulary also comes from the normal logs, so words never seen there are ignored.
        tfidf_matrix = vectorizer.transform(log_df["log_template"])
        labels = kmeans.predict(tfidf_matrix)
        distances = compute_centroid_distances(tfidf_matrix, labels, kmeans.cluster_centers_)
        # Threshold = the distance only outlier_percent of NORMAL logs go beyond.
        distance_threshold = float(np.percentile(train_distances, 100.0 - outlier_percent))
        anomaly_mask = distances > distance_threshold
    else:
        labels, distances = train_labels, train_distances
        # Select the globally furthest N% — this is the actual anomaly decision boundary.
        outlier_count = max(1, int(np.ceil(n_samples * (outlier_percent / 100.0))))
        distance_threshold = float(np.sort(distances)[::-1][outlier_count - 1])
        # Using >= means identical lines that tie exactly on the threshold are either all
        # flagged or all not. Picking only some of them (what argsort did before) was
        # arbitrary — 53 of 142 identical SMF lines got flagged just by sort order.
        anomaly_mask = distances >= distance_threshold

    result_df = log_df.copy()
    result_df["cluster"] = labels
    result_df["centroid_distance"] = distances
    result_df["is_anomaly"] = anomaly_mask

    anomalies_df = (
        result_df[result_df["is_anomaly"]]
        .sort_values("centroid_distance", ascending=False)
        .copy()
    )

    # --- Threat categorisation (post-processing, ONLY on the already-flagged anomalies) ---
    if not anomalies_df.empty:
        anomalies_df["proposed_threat_category"] = anomalies_df["raw_log"].apply(classify_threat)
        anomalies_df["high_risk"] = anomalies_df["proposed_threat_category"].isin(HIGH_RISK_CATEGORIES)
        anomalies_df["threat_display"] = anomalies_df.apply(
            lambda r: f"{r['proposed_threat_category']}{' [HIGH RISK - REQUIRES ANALYST CONFIRMATION]' if r['high_risk'] else ''}",
            axis=1,
        )
    else:
        # Ensure columns exist even when empty for downstream code that expects them
        anomalies_df["proposed_threat_category"] = []
        anomalies_df["high_risk"] = []
        anomalies_df["threat_display"] = []

    # Separate display list — re-ranked for the report, doesn't change the CSV anomaly set.
    display_anomalies_df = prepare_display_anomalies(anomalies_df)
    cluster_sizes = result_df["cluster"].value_counts().sort_index()

    mode_note = (
        f"beyond the normal-log {100 - outlier_percent:g}th percentile"
        if trained_on_baseline
        else f"{outlier_percent}% furthest from centroid"
    )
    print(
        f"Micro-clustering complete: k={effective_clusters} | "
        f"Outliers flagged: {len(anomalies_df):,} ({mode_note})"
    )

    return DetectionResult(
        anomalies_df=anomalies_df,
        cluster_sizes=cluster_sizes,
        result_df=result_df,
        n_clusters_used=effective_clusters,
        outlier_count=len(anomalies_df),
        outlier_percent=outlier_percent,
        distance_threshold=distance_threshold,
        max_distance=float(distances.max()),
        mean_distance=float(distances.mean()),
        display_anomalies_df=display_anomalies_df,
        trained_on_baseline=trained_on_baseline,
        baseline_events=n_train if trained_on_baseline else 0,
    )
