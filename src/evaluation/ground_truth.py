"""
Ground-truth labels for the benchmark.

To calculate Accuracy / Precision / Recall / F1 I need to know which log events really
are anomalous. My logs don't come with labels, so I support two ways of labelling them:

1. Per line, from the attack timeline (the default, used for my results). I wrote down
   exactly when I started and stopped the attack (attack_timeline.log), so every line
   can be labelled on its own. See label_by_attack_window at the bottom.
2. Per template, from a CSV (run_benchmark.py --labels templates). Every unique template
   in the attack captures gets one row and a 1/0 label that applies to all its lines.
   Each row gets a *suggested* label from a simple rule (see suggest_label); a label
   typed into the 'label' column always wins, and the report warns while any rows are
   still running on suggestions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.summary_report import extract_log_level

LABEL_COLUMNS = [
    "network_function",
    "log_template",
    "log_level",
    "attack_count",
    "baseline_count",
    "suggested_label",
    "suggested_reason",
    "label",
    "notes",
    "example_raw_log",
]

# Log levels that OpenAirInterface only uses when something has actually gone wrong.
FAILURE_LEVELS = {"error", "fatal", "critical"}


def suggest_label(log_level: str, baseline_count: int) -> tuple[int, str]:
    """
    My provisional rule for which templates are anomalous.

    1. The template never appears in the normal (benign) capture for the same
       network function, so it's behaviour the core didn't show when nobody was attacking it.
    2. OR the line is logged at error/fatal/critical level.

    Rule 2 overlaps with what the keyword filter looks for, so it slightly favours
    the keyword baseline. That's another reason these must be checked by hand
    before the numbers go into the report.
    """
    if baseline_count == 0:
        return 1, "template never seen in the normal capture"
    if log_level in FAILURE_LEVELS:
        return 1, f"logged at [{log_level}] level"
    return 0, "seen in the normal capture and not an error"


def build_label_table(attack_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Create one label row per (network function, template) found in the attack logs."""
    baseline_counts = (
        baseline_df.groupby(["network_function", "log_template"]).size().rename("baseline_count")
    )

    grouped = attack_df.groupby(["network_function", "log_template"])
    table = grouped.size().rename("attack_count").to_frame()
    table["example_raw_log"] = grouped["raw_log"].first()
    table = table.join(baseline_counts, how="left").fillna({"baseline_count": 0}).reset_index()
    table["baseline_count"] = table["baseline_count"].astype(int)
    table["log_level"] = table["example_raw_log"].apply(extract_log_level)

    suggestions = table.apply(
        lambda row: suggest_label(row["log_level"], row["baseline_count"]), axis=1
    )
    table["suggested_label"] = [s[0] for s in suggestions]
    table["suggested_reason"] = [s[1] for s in suggestions]
    table["label"] = ""
    table["notes"] = ""

    table = table.sort_values(
        ["network_function", "suggested_label", "attack_count"], ascending=[True, False, False]
    )
    return table[LABEL_COLUMNS]


def merge_existing_labels(new_table: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    """
    Keep any labels and notes I already filled in by hand when the file gets rebuilt.

    Rows are matched on (network_function, log_template). If I change the regex
    cleaning in log_parser.py some templates will change and lose their old label,
    so the build script tells me how many were carried over.
    """
    if not label_path.is_file():
        return new_table

    old = pd.read_csv(label_path, dtype=str, keep_default_na=False)
    old = old[["network_function", "log_template", "label", "notes"]]
    merged = new_table.drop(columns=["label", "notes"]).merge(
        old, on=["network_function", "log_template"], how="left"
    )
    merged[["label", "notes"]] = merged[["label", "notes"]].fillna("")
    return merged[LABEL_COLUMNS]


def load_labels(label_path: Path) -> pd.DataFrame:
    """
    Read the label file and work out the label the benchmark should use per template.

    'final_label' = my manual 'label' if I filled it in, otherwise the suggestion.
    'is_manual' records which one was used so the report can say how provisional it is.
    """
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Label file not found: {label_path}\n"
            "Run 'python scripts/build_ground_truth.py' first to create it."
        )

    labels = pd.read_csv(label_path, dtype=str, keep_default_na=False)
    manual = labels["label"].str.strip()
    labels["is_manual"] = manual.isin(["0", "1"])
    labels["final_label"] = manual.where(labels["is_manual"], labels["suggested_label"]).astype(int)
    return labels[["network_function", "log_template", "final_label", "is_manual"]]


def attach_labels(log_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Give every log line the label of its template.

    A line whose template isn't in the label file at all (e.g. the attack logs changed
    after the file was built) can't be scored, so I stop with a clear message instead
    of silently guessing.
    """
    labelled = log_df.merge(labels, on=["network_function", "log_template"], how="left")
    missing = labelled["final_label"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} log lines have templates that aren't in the label file. "
            "Re-run 'python scripts/build_ground_truth.py' to add them."
        )
    labelled["final_label"] = labelled["final_label"].astype(int)
    labelled["is_manual"] = labelled["is_manual"].astype(bool)
    return labelled


# ---------------------------------------------------------------------------
# Per-line labels from the attack timeline
# ---------------------------------------------------------------------------
# Template labels have one weakness: the same template (e.g. "Received Registration
# Request") is logged for my real UE AND for every fake UE in the storm, so one label
# can't be right for both. Because I recorded exactly when I ran the attack
# (attack_timeline.log), I can label every line on its own instead.

# A template counts as "produced by the attack" if at least this share of its lines
# fall inside the attack window. Background lines (NRF heartbeats etc.) tick along the
# whole capture, so only a small share of them lands in a ~1 minute window.
ATTACK_TEMPLATE_SHARE = 0.5


def load_attack_windows(timeline_path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Read the ATTACK_<name>_START / ATTACK_<name>_STOP marks from attack_timeline.log.

    Each line looks like 'ATTACK_REG_STORM_START,2026-09-30 15:31:16.776,UTC ...'.
    The second field is local time, which matches the timestamps in the OAI logs.
    """
    starts: dict[str, pd.Timestamp] = {}
    windows = []
    for line in timeline_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2 or not parts[0].startswith("ATTACK_"):
            continue
        mark, when = parts[0], pd.Timestamp(parts[1])
        if mark.endswith("_START"):
            starts[mark[: -len("_START")]] = when
        elif mark.endswith("_STOP") and mark[: -len("_STOP")] in starts:
            windows.append((starts.pop(mark[: -len("_STOP")]), when))
    if not windows:
        raise ValueError(f"No ATTACK_*_START / ATTACK_*_STOP pairs found in {timeline_path}")
    return windows


def label_by_attack_window(
    log_df: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    """
    Label each line 1 if the attack produced it, otherwise 0.

    A line is an attack line when:
      1. it was logged inside one of the attack windows, AND
      2. its template is one the attack produced: most of that template's lines
         (per network function) fall inside the windows.
    Rule 2 stops background lines that just happen to tick during the window
    (heartbeats, NRF updates) from being counted as attack.
    """
    labelled = log_df.copy()
    timestamps = pd.to_datetime(labelled["raw_log"].str.extract(r"^\[([^\]]+)\]")[0])
    in_window = pd.Series(False, index=labelled.index)
    for start, stop in windows:
        in_window |= timestamps.between(start, stop)

    window_share = in_window.groupby(
        [labelled["network_function"], labelled["log_template"]]
    ).transform("mean")
    labelled["final_label"] = (in_window & (window_share >= ATTACK_TEMPLATE_SHARE)).astype(int)
    labelled["is_manual"] = False  # rule-based, not typed in by hand
    return labelled
