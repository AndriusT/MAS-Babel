"""Core data models — topology-agnostic."""

from typing import Dict, List
from pydantic import BaseModel


class TopologyResult(BaseModel):
    """Internal result returned by every topology run() function.

    Attributes:
        response_text: The final response text to be parsed into DesignOutput.
        messages: Flat list of dicts with "source" and "content" keys, used for logging.
        rounds_used: Number of Architect turns consumed (for logging/analysis).
    """

    response_text: str
    messages: List[Dict[str, str]] = []
    rounds_used: int = 1


__all__ = ["TopologyResult"]

