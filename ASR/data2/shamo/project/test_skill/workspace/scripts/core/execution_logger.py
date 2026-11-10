"""Unified execution logger for all CLI scripts.

Appends each script execution result to ``logs/execution_log.jsonl`` as a
single JSON line with timestamp, script name, and output payload.

Usage in CLI scripts:

    from scripts.core.execution_logger import log_execution

    def main():
        ...
        output = {"status": "success", ...}
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\\n")
        log_execution("script_name", output)

Log format (each line in logs/execution_log.jsonl):
    {"timestamp": "2026-05-21T10:00:00", "script": "run_augmentation", "output": {...}}
"""

import json
import os
from datetime import datetime
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "execution_log.jsonl"


def log_execution(script_name: str, output: dict) -> None:
    """Append an execution record to the unified log file.

    Parameters
    ----------
    script_name : str
        Name of the script (e.g. "run_augmentation", "analyze_results").
    output : dict
        The JSON output that was printed to stdout.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": script_name,
        "output": output,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
