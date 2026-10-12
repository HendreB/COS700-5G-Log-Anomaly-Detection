"""
Phase 2 — Log preprocessing for my COS700 5G anomaly detection prototype.

The OpenAirInterface logs are semi-structured: timestamps, IMSI numbers, and IPs
change on every line but the underlying message is often the same. I strip those
dynamic parts out with regex so the ML model sees stable "templates" instead of
raw noisy text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Only lines matching this pattern count as real log events — filters out ASCII banners
# and startup art (especially in the UPF logs).
# Side effect I need to remember: continuation lines of multi-line messages (e.g. the
# indented "criticality: 0 (reject)" lines under NGAP dumps) have no timestamp, so they
# are dropped too. Only the first line of each message becomes an event.
LOG_LINE_PATTERN = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\]"
)

# Strip ANSI colour codes (present in some UPF startup banners/lines).
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def clean_log_line(log_line: str) -> str:
    """
    Strip dynamic values from a log line and return a static template.

    Handles OpenAirInterface format:
        [2026-08-18 11:04:21.264] [amf_app] [info] message...
    """
    line = ANSI_ESCAPE_PATTERN.sub("", log_line).strip()
    if not line:
        return line

    # Each substitution replaces one type of dynamic value with a placeholder token.
    # Order matters — I do timestamps and paths before generic numbers so I don't
    # accidentally break patterns like IP addresses or IMSI strings.
    # Bracketed OpenAir timestamps: [2026-08-18 11:04:21.264]
    line = re.sub(
        r"\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\]",
        "<TIMESTAMP>",
        line,
    )
    # Also handle ISO format in case I add synthetic test logs later.
    line = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z",
        "<TIMESTAMP>",
        line,
    )
    line = re.sub(r"/[\w./-]+", "<FILE_PATH>", line)
    # IPv4 addresses
    line = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP_ADDRESS>", line)
    # IMSI / SUPI identifiers
    line = re.sub(r"imsi-\d+", "<IMSI>", line, flags=re.IGNORECASE)
    # Hexadecimal codes (e.g. 0xdeadbeef, 0x7a)
    line = re.sub(r"0x[0-9a-fA-F]+", "<HEX_CODE>", line)
    # UUIDs
    line = re.sub(
        r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
        "<UUID>",
        line,
        flags=re.IGNORECASE,
    )
    # Long hex strings WITHOUT a 0x prefix (auth vectors, keys, message dumps like
    # "Data 57ecc439c69e11c7..."). Without this every one became its own unique template
    # and the model flagged them just for being random. It needs at least one letter a-f
    # and one digit so plain numbers (IMSIs, ports) still become <NUMERIC_ID> below.
    # Must run after the UUID rule, otherwise it would eat the UUID's first block.
    line = re.sub(
        r"\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b",
        "<HEX_CODE>",
        line,
    )
    # Short hex values that change on every line but are too short for the rule above
    # ("pfcp_tx_id 1A", "stream_cipher.message 1b"). I only match them after these known
    # fields, because a general short-hex rule would also hit fixed config like "SD 00007b".
    line = re.sub(
        r"\b(pfcp_tx_id|stream_cipher\.message)\s+[0-9a-fA-F]+\b",
        r"\1 <HEX_CODE>",
        line,
    )
    # Uptime durations in the UPF startup table ("0d 0h 40m")
    line = re.sub(r"\b\d+d \d+h \d+m\b", "<DURATION>", line)
    # Remaining standalone numbers (IDs, counters, ports)
    line = re.sub(r"\b\d+\b", "<NUMERIC_ID>", line)
    # Normalize whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line


def infer_network_function(filename: str) -> str:
    """Map a log filename to its 5G network function."""
    name = filename.lower()
    if "amf" in name:
        return "AMF"
    if "smf" in name:
        return "SMF"
    if "upf" in name:
        return "UPF"
    return "UNKNOWN"


def load_and_preprocess_logs(log_file_path: str | Path) -> pd.DataFrame:
    """
    Load a raw log file, extract static templates, and return a DataFrame.

    Non-log lines (banners, blank lines, config preamble) are skipped.
    """
    log_path = Path(log_file_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    raw_logs: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            cleaned = ANSI_ESCAPE_PATTERN.sub("", line).strip()
            if cleaned and LOG_LINE_PATTERN.match(cleaned):
                raw_logs.append(cleaned)

    df = pd.DataFrame({"raw_log": raw_logs})
    df["log_template"] = df["raw_log"].apply(clean_log_line)
    # Track where each line came from — useful when running --log all for the combined report.
    df["source_file"] = log_path.name
    df["network_function"] = infer_network_function(log_path.name)
    print(f"Loaded and preprocessed {len(df)} log entries from '{log_path.name}'.")
    return df


def load_all_logs(data_dir: Path) -> pd.DataFrame:
    """Load and combine all .txt log files — used for the unified 5G core analysis."""
    log_files = list_available_logs(data_dir)
    if not log_files:
        raise FileNotFoundError(f"No .txt log files found in '{data_dir}'.")

    frames: list[pd.DataFrame] = []
    for path in log_files:
        frames.append(load_and_preprocess_logs(path))

    combined = pd.concat(frames, ignore_index=True)
    print(
        f"Combined dataset: {len(combined):,} events across "
        f"{len(log_files)} source artefact(s)."
    )
    return combined


def list_available_logs(data_dir: Path) -> list[Path]:
    """Return sorted .txt log files found in the data directory."""
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.txt"))


def is_attack_log(path: Path) -> bool:
    """Attack captures are named *_attack_logs.txt; everything else counts as normal traffic."""
    return path.name.endswith("_attack_logs.txt")


def load_normal_logs(data_dir: Path, network_functions: set[str]) -> pd.DataFrame:
    """
    Load the normal (non-attack) captures for the given network functions.

    Used when the model learns "normal behaviour" from clean logs first and then
    scores a different log against it — e.g. AMF attack logs vs the AMF normal log.
    """
    paths = [
        p for p in list_available_logs(data_dir)
        if not is_attack_log(p) and infer_network_function(p.name) in network_functions
    ]
    if not paths:
        raise FileNotFoundError(
            f"No normal log files for {sorted(network_functions)} found in '{data_dir}'."
        )
    return pd.concat([load_and_preprocess_logs(p) for p in paths], ignore_index=True)
