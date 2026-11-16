"""Shared code execution helper for topology functions.

Constructing a :class:`CodeExecutorAgent` is identical across all topologies.
This module centralises that so timeouts and work_dir logic live in one place.

Usage::

    from ._executor import make_code_executor_agent, contains_code_block

    if execute_code:
        code_runner = make_code_executor_agent(work_dir)
"""

import re
from pathlib import Path
from typing import Optional, Sequence

from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage


def make_code_executor_agent(work_dir: str, sources: Optional[Sequence[str]] = None):
    """Return a :class:`CodeExecutorAgent` backed by a local subprocess executor.

    The *work_dir* is created if it does not exist.  Each parallel expert should
    get its own subdirectory to avoid concurrent write collisions.

    Args:
        work_dir: Path to the working directory for code execution.
        sources: Only execute code from messages of these agents (e.g. the
            generator), so code that a critic quotes is never run.  ``None``
            executes code from every message.

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
    return CodeExecutorAgent(name="CodeRunner", code_executor=executor, sources=sources)


def contains_code_block(text: str) -> bool:
    """Return True if *text* contains a fenced code block (``` ... ```)."""
    return "```" in text


_DONE = re.compile(r"\bDONE\b")
_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.DOTALL)


class DoneTermination(TerminationCondition):
    """Stop an agent + CodeRunner chat once *agent_name* says ``DONE``.

    Only messages from *agent_name* are checked, for the whole word ``DONE``
    outside code blocks (``DONE = True`` inside code is not a declaration).
    If that message also contains a code block, the chat stops after the next
    message (the CodeRunner's output) so the final code is always executed.
    """

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._terminated = False
        self._stop_after_next = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> Optional[StopMessage]:
        for message in messages:
            if not isinstance(message, BaseChatMessage):
                continue
            if self._stop_after_next:
                return self._stop("DONE (after executing the final code)")
            if message.source != self._agent_name:
                continue
            text = message.to_text()
            if _DONE.search(_CODE_BLOCK.sub("", text)):
                if not contains_code_block(text):
                    return self._stop("DONE")
                self._stop_after_next = True
        return None

    def _stop(self, reason: str) -> StopMessage:
        self._terminated = True
        return StopMessage(content=f"{self._agent_name} said {reason}", source="DoneTermination")

    async def reset(self) -> None:
        self._terminated = False
        self._stop_after_next = False


__all__ = ["make_code_executor_agent", "contains_code_block", "DoneTermination"]
