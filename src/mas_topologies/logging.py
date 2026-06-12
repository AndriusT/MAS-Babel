"""Append-only JSONL session logging."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from .models import TopologyResult


def log_session(
    log_path: Optional[str],
    topology: str,
    result: TopologyResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON line per session to a JSONL log file.

    No-op when *log_path* is ``None``.  Task-specific fields (dataset name,
    iteration, etc.) can be passed via *metadata* and are merged into the log entry.
    """
    if log_path is None:
        return

    messages = []
    round_num = 0
    for msg in result.messages:
        source = msg["source"]
        content = msg["content"]
        if source == "Architect":
            round_num += 1
            messages.append({"source": "Architect", "round": round_num, "content": content})
        elif source in ("Reviewer", "Adversary", "Judge"):
            approved = content.strip() in ("APPROVED", "ACCEPTED")
            messages.append(
                {
                    "source": source,
                    "round": round_num,
                    "approved": approved,
                    "content": content,
                }
            )

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "topology": topology,
        "rounds_used": result.rounds_used,
        "approved": any(m.get("approved", False) for m in messages),
        "messages": messages,
        **(metadata or {}),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = ["log_session"]
