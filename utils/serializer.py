# ============================================================
#  utils/serializer.py  –  JSON record output & JSONL logging
# ============================================================

import json
import os
from datetime import datetime, timezone
from typing import Optional

from config import LOG_PATH
from utils.decision import DecisionResult


def _ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def build_record(track_id: int, result: DecisionResult) -> dict:
    """Assemble the final JSON record for one Track ID."""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "track_id":  track_id,
        "gender":    result.gender,
        "age_range": result.age_range,
    }


def write_record(record: dict):
    """Append one JSON record to the JSONL log file."""
    _ensure_log_dir()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def print_record(record: dict, result: Optional[DecisionResult] = None):
    """
    Pretty-print the record to terminal with debug extras.
    """
    sep = "─" * 54
    print(f"\n{sep}")
    print(f"  ✔  Track ID   : {record['track_id']}")
    print(f"     Timestamp  : {record['timestamp']}")
    print(f"     Gender     : {record['gender']}")
    print(f"     Age Range  : {record['age_range']}")
    if result:
        print(f"  ── debug ──────────────────────────────────────")
        print(f"     Source     : {result.source}")
        print(f"     P(Female)  : {result.gender_score:.3f}")
        print(f"     Age (raw)  : {result.age_raw:.1f}")
    print(f"{sep}\n")