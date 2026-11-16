"""Core data models — topology-agnostic."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TopologyResult(BaseModel):
    """Internal result returned by every topology run() function.

    Attributes:
        response_text: The full final message of the run — the last generator
            (Architect) message, or the Judge's message for ``parallel_ensemble``.
        messages: Flat list of dicts with "source" and "content" keys, used for logging.
            Messages produced by a model call also carry "prompt_tokens" and
            "completion_tokens".
        rounds_used: Number of Architect turns consumed (for logging/analysis).
        final_answer: The answer extracted from *response_text* (the text after
            ``FINAL ANSWER:``).  Set by :class:`~mas_topologies.agent.TopologyAgent`.
        final_answer_tagged: ``False`` when the tag was missing and
            *final_answer* fell back to the whole of *response_text*.
        usage: Token usage of the run, ``{"total": {...}, "by_source": {...}}``
            with ``prompt_tokens``, ``completion_tokens`` and ``calls``.  Set by
            :class:`~mas_topologies.agent.TopologyAgent`.
        info: Topology-specific details of the run (e.g. ``debate``: votes,
            winner, tie, consensus round, answers per round).  Logged as is.
    """

    response_text: str
    messages: List[Dict[str, Any]] = []
    rounds_used: int = 1
    final_answer: Optional[str] = None
    final_answer_tagged: bool = False
    usage: Dict[str, Any] = {}
    info: Dict[str, Any] = {}


__all__ = ["TopologyResult"]

