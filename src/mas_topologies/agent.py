"""TopologyAgent — generic multi-agent topology dispatcher."""

import os
from datetime import datetime
from typing import Any, Dict, Optional

from .logging import log_session
from .models import TopologyResult
from .prompts import PromptSet
from .topologies import (
    run_adversarial,
    run_parallel_ensemble,
    run_peer_review,
    run_self_reflection,
    run_single,
)

VALID_TOPOLOGIES = (
    "single",
    "peer_review",
    "adversarial",
    "parallel_ensemble",
    "self_reflection",
)


class TopologyAgent:
    """Generic multi-agent topology dispatcher.

    Accepts a :class:`PromptSet` at construction time and a plain context string
    at run time.  All task-specific logic (context building, response parsing)
    lives outside this class, keeping the topology machinery domain-agnostic.

    Supported topologies (``topology`` parameter):

    * ``"single"``            -- single LLM call (fastest)
    * ``"peer_review"``       -- primary + critic iterate (max 4 rounds)
    * ``"adversarial"``       -- primary + adversary iterate (max 4 rounds)
    * ``"parallel_ensemble"`` -- N agents in parallel, judge synthesises
    * ``"self_reflection"``   -- agent proposes then self-critiques (max 3 rounds)

    When *log_dir* is provided each session is appended as one JSON line to
    ``<log_dir>/session_<timestamp>.jsonl``.
    """

    def __init__(
        self,
        model_client,
        topology: str = "single",
        prompts: Optional[PromptSet] = None,
        log_dir: Optional[str] = None,
        n_experts: int = 3,
        judge_model_client=None,
        execute_code: bool = False,
        work_dir: str = "code_workspace",
        max_rounds: int = 3,
    ):
        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"Unknown topology {topology!r}. Valid options: {VALID_TOPOLOGIES}"
            )
        if prompts is None:
            raise ValueError(
                "A PromptSet must be provided. "
                "Build one with PromptSet(system_message='...') or use a task-specific preset."
            )

        self.model_client = model_client
        self.topology = topology
        self.prompts = prompts
        self.n_experts = n_experts
        self.judge_model_client = judge_model_client or model_client
        self.execute_code = execute_code
        self.work_dir = work_dir
        self.max_rounds = max_rounds

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"session_{ts}.jsonl")
            print(f"[TopologyAgent] Session log: {self._log_path}")
        else:
            self._log_path = None

    async def run(
        self,
        context: str,
        interactive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TopologyResult:
        """Run the configured topology and return a :class:`TopologyResult`.

        Args:
            context:     Pre-built context string passed to the primary agent.
            interactive: Enable console mode (honoured by ``single`` only).
            metadata:    Optional dict merged into the JSONL log entry.

        Returns:
            :class:`TopologyResult` with ``response_text``, ``messages``,
            and ``rounds_used``.
        """
        print("\n" + "=" * 80)
        print(f"[TopologyAgent] topology={self.topology}")
        print("=" * 80)

        p = self.prompts

        # Build effective system message: append code_execution_hint when active
        effective_system = p.system_message
        if self.execute_code and p.code_execution_hint:
            effective_system = effective_system + "\n\n" + p.code_execution_hint

        if self.topology == "single":
            result = await run_single(
                context, self.model_client, effective_system, interactive,
                execute_code=self.execute_code, work_dir=self.work_dir,
            )
        elif self.topology == "peer_review":
            result = await run_peer_review(
                context, self.model_client, effective_system, p.critic_message,
                execute_code=self.execute_code, work_dir=self.work_dir,
            )
        elif self.topology == "adversarial":
            result = await run_adversarial(
                context, self.model_client, effective_system, p.adversary_message,
                execute_code=self.execute_code, work_dir=self.work_dir,
            )
        elif self.topology == "parallel_ensemble":
            result = await run_parallel_ensemble(
                context,
                self.model_client,
                effective_system,
                judge_message=p.judge_message,
                judge_model_client=self.judge_model_client,
                n_experts=self.n_experts,
                execute_code=self.execute_code,
                work_dir=self.work_dir,
            )
        elif self.topology == "self_reflection":
            result = await run_self_reflection(
                context, self.model_client, effective_system, p.reflection_prompt,
                max_rounds=self.max_rounds,
                execute_code=self.execute_code, work_dir=self.work_dir,
            )
        else:
            raise ValueError(f"Unknown topology: {self.topology!r}")

        log_session(self._log_path, self.topology, result, metadata)
        return result


__all__ = ["TopologyAgent", "VALID_TOPOLOGIES"]

