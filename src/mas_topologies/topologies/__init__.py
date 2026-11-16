"""Topology modules. Each exposes ``async def run(...) -> TopologyResult``."""

from .single import run as run_single
from .peer_review import run as run_peer_review
from .adversarial import run as run_adversarial
from .parallel_ensemble import run as run_parallel_ensemble
from .self_reflection import run as run_self_reflection
from .debate import run as run_debate
from ._executor import make_code_executor_agent, contains_code_block

__all__ = [
    "run_single",
    "run_peer_review",
    "run_adversarial",
    "run_parallel_ensemble",
    "run_self_reflection",
    "run_debate",
    "make_code_executor_agent",
    "contains_code_block",
]
