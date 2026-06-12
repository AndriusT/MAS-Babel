"""Shared code execution helper for topology functions.

Constructing a :class:`CodeExecutorAgent` is identical across all topologies.
This module centralises that so timeouts and work_dir logic live in one place.

Usage::

    from ._executor import make_code_executor_agent, contains_code_block

    if execute_code:
        code_runner = make_code_executor_agent(work_dir)
"""

from pathlib import Path


def make_code_executor_agent(work_dir: str):
    """Return a :class:`CodeExecutorAgent` backed by a local subprocess executor.

    The *work_dir* is created if it does not exist.  Each parallel expert should
    get its own subdirectory to avoid concurrent write collisions.

    Args:
        work_dir: Path to the working directory for code execution.

    Returns:
        A configured ``CodeExecutorAgent`` ready to be inserted into a group chat.
    """
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    executor = LocalCommandLineCodeExecutor(
        work_dir=work_dir,
        timeout=120,  # seconds per execution; increase for heavier tasks
    )
    return CodeExecutorAgent(name="CodeRunner", code_executor=executor)


def contains_code_block(text: str) -> bool:
    """Return True if *text* contains a fenced code block (``` ... ```)."""
    return "```" in text


__all__ = ["make_code_executor_agent", "contains_code_block"]
