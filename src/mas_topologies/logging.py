"""Append-only JSONL session logging."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from .models import TopologyResult
from .protocol import is_approval

# Roles that give a VERDICT; their log entries carry an "approved" flag
_CRITIC_SOURCES = ("Reviewer", "Adversary", "Reflection")
_TOKEN_KEYS = ("prompt_tokens", "completion_tokens")


def log_session(
    log_path: Optional[str],
    topology: str,
    result: TopologyResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON line per session to a JSONL log file.

    No-op when *log_path* is ``None``.  Task-specific fields (dataset name,
    iteration, etc.) can be passed via *metadata* and are merged into the log entry.
    Every message is logged; critic messages also carry an ``approved`` flag.
    """
    if log_path is None:
        return

    messages = []
    round_num = 0
    for msg in result.messages:
        source = msg["source"]
        content = msg["content"]
        # Token counts of model-call messages (CodeRunner output has none)
        tokens = {k: msg[k] for k in _TOKEN_KEYS if k in msg}
        if source == "Architect":
            round_num += 1
            messages.append({"source": "Architect", "round": round_num, "content": content, **tokens})
        elif source in _CRITIC_SOURCES:
            approved = is_approval(content)  # same rule that stops the run
            messages.append(
                {
                    "source": source,
                    "round": round_num,
                    "approved": approved,
                    "content": content,
                    **tokens,
                }
            )
        else:
            # CodeRunner, Judge, Architect_N, Debater_N, ... — keep everything.  Topologies
            # whose messages carry their own round number (debate) keep it.
            messages.append({"source": source, "round": msg.get("round", round_num),
                             "content": content, **tokens})

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "topology": topology,
        "rounds_used": result.rounds_used,
        "approved": any(m.get("approved", False) for m in messages),
        "final_answer": result.final_answer,
        "final_answer_tagged": result.final_answer_tagged,
        "usage": result.usage,
        "info": result.info,
        "messages": messages,
        **(metadata or {}),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = ["log_session"]
