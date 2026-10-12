"""
Phase 4 — Traditional keyword filtering baseline (research question 2.1.3).

This is the "old way" I'm comparing my prototype against. SIEM tools and forensic
investigators usually just grep the logs for alarm words like "error" or "fail"
and treat every hit as suspicious. I copy that behaviour here as fairly as I can:
one case-insensitive regex over the raw log line, no ML and no preprocessing.

Nothing in here touches the TF-IDF model. The two methods only meet in benchmark.py.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pandas as pd

# The alarm words a typical SIEM correlation rule / analyst grep would look for.
# I keep them in one list so I can quote it exactly in the report, and so it's easy
# to test a stricter or looser rule later. \b word boundaries stop "fail" matching
# inside unrelated words.
ALERT_KEYWORDS = [
    r"error",
    r"fail(?:s|ed|ure)?",
    r"fatal",
    r"critical",
    r"warn(?:ing)?",
    r"reject(?:s|ed)?",
    r"denied",
    r"unauthori[sz]ed",
    r"invalid",
    r"refused",
    r"time(?:d)?\s?out",
    r"abort(?:s|ed)?",
    r"exception",
    r"attack",
    r"flood",
    r"drop(?:s|ped)?",
]

KEYWORD_PATTERN = re.compile(r"\b(?:" + "|".join(ALERT_KEYWORDS) + r")\b", re.IGNORECASE)


@dataclass
class KeywordResult:
    """What the keyword filter returns — same idea as DetectionResult for the ML model."""

    flags: pd.Series            # True where at least one alarm word was found
    matched_keyword: pd.Series  # The first alarm word that matched (empty if none)
    runtime_seconds: float      # Time spent filtering, for the efficiency comparison


def keyword_filter(log_df: pd.DataFrame) -> KeywordResult:
    """
    Flag every log line that contains one of the alarm keywords.

    I run it on 'raw_log' (not the cleaned template) because a real analyst would
    grep the original file, not my preprocessed version.
    """
    start = time.perf_counter()
    matches = log_df["raw_log"].str.extract(f"({KEYWORD_PATTERN.pattern})", flags=re.IGNORECASE)[0]
    flags = matches.notna()
    runtime = time.perf_counter() - start

    return KeywordResult(
        flags=flags,
        matched_keyword=matches.fillna("").str.lower(),
        runtime_seconds=runtime,
    )
