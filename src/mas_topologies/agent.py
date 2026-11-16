"""TopologyAgent — generic multi-agent topology dispatcher."""

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import log_session
from .models import TopologyResult
from .prompts import PromptSet
from .protocol import (
    DONE_FINAL_ANSWER_INSTRUCTION,
    extract_final_answer,
    normalize_answer,
    with_final_answer_instruction,
)
from .topologies._messages import summarize_usage
from .topologies import (
    run_adversarial,
    run_debate,
    run_parallel_ensemble,
    run_peer_review,
    run_self_reflection,
    run_single,
)
from .topologies.debate import AGGREGATIONS

VALID_TOPOLOGIES = (
    "single",
    "peer_review",
    "adversarial",
    "parallel_ensemble",
    "self_reflection",
    "debate",
)


class TopologyAgent:
    """Generic multi-agent topology dispatcher.

    Accepts a :class:`PromptSet` at construction time and a plain context string
    at run time.  All task-specific logic (context building, response parsing)
    lives outside this class, keeping the topology machinery domain-agnostic.

    Supported topologies (``topology`` parameter):

    * ``"single"``            -- single LLM call (fastest)
    * ``"peer_review"``       -- primary + critic iterate (default max 4 rounds)
    * ``"adversarial"``       -- primary + adversary iterate (default max 4 rounds)
    * ``"parallel_ensemble"`` -- N agents in parallel, judge synthesises
    * ``"self_reflection"``   -- agent proposes then self-critiques (default max 3 rounds)
    * ``"debate"``            -- N debaters see each other's answers and revise
      (default max 3 rounds), then majority vote (*aggregation="vote"*) or a
      judge (``"judge"``); ``max_rounds=1`` + vote is self-consistency

    *max_rounds* is the maximum number of primary-agent turns (proposals and
    revisions) in ``peer_review``, ``adversarial``, ``self_reflection`` and
    ``debate`` (turns per debater).  ``None`` keeps each topology's default
    shown above.  ``debate`` uses *n_experts* debaters, stops early once all
    agree (*stop_on_consensus*), and compares answers with *answer_normalizer*.

    Each role can use its own model: *model_client* drives the generator
    (Architect / experts), while *critic_model_client* (Reviewer),
    *adversary_model_client* (Adversary) and *judge_model_client* (Judge) each
    fall back to *model_client* when not given.  *expert_model_clients* (a list)
    gives the parallel experts / debaters different models: agent *i* uses
    ``expert_model_clients[i % len(expert_model_clients)]``.

    Every run's ``result.final_answer`` comes from *answer_extractor* (default
    :func:`~mas_topologies.protocol.extract_final_answer`) applied to the final
    message: the text after ``FINAL ANSWER:``, or the whole message with
    ``final_answer_tagged=False`` if the tag is missing.

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
        max_rounds: Optional[int] = None,
        critic_model_client=None,
        adversary_model_client=None,
        answer_extractor: Callable[[str], Tuple[str, bool]] = extract_final_answer,
        expert_model_clients: Optional[List] = None,
        aggregation: str = "vote",
        answer_normalizer: Callable[[str], str] = normalize_answer,
        stop_on_consensus: bool = True,
    ):
        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"Unknown topology {topology!r}. Valid options: {VALID_TOPOLOGIES}"
            )
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if expert_model_clients is not None and len(expert_model_clients) == 0:
            raise ValueError("expert_model_clients must be a non-empty list (or None)")
        if prompts is None:
            raise ValueError(
                "A PromptSet must be provided. "
                "Build one with PromptSet(system_message='...') or use a task-specific preset."
            )

        self.model_client = model_client
        self.topology = topology
        self.prompts = prompts
        self.n_experts = n_experts
        # Per-role models; each falls back to the generator's model
        self.judge_model_client = judge_model_client or model_client
        self.critic_model_client = critic_model_client or model_client
        self.adversary_model_client = adversary_model_client or model_client
        self.expert_model_clients = list(expert_model_clients) if expert_model_clients else None
        self.execute_code = execute_code
        self.work_dir = work_dir
        self.max_rounds = max_rounds
        self.answer_extractor = answer_extractor
        self.aggregation = aggregation
        self.answer_normalizer = answer_normalizer
        self.stop_on_consensus = stop_on_consensus

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
            ``rounds_used``, ``final_answer``, ``final_answer_tagged`` and ``usage``.
        """
        print("\n" + "=" * 80)
        print(f"[TopologyAgent] topology={self.topology}")
        print("=" * 80)

        p = self.prompts

        # Build effective system message: ask for a FINAL ANSWER tag, and append
        # code_execution_hint when active
        effective_system = with_final_answer_instruction(p.system_message, p.final_answer_hint)
        if self.execute_code and p.code_execution_hint:
            effective_system = effective_system + "\n\n" + p.code_execution_hint
        if self.execute_code and p.final_answer_hint:
            # The run ends on the DONE message, so that message must hold the answer
            effective_system = effective_system + "\n\n" + DONE_FINAL_ANSWER_INSTRUCTION

        # Options for the iterative (critic-based) topologies.  Only override a
        # topology's own default round limit when one was given.
        iter_kw = {} if self.max_rounds is None else {"max_rounds": self.max_rounds}
        iter_kw["verdict_instruction"] = p.verdict_hint

        if self.topology == "single":
            result = await run_single(
                context, self.model_client, effective_system, interactive,
                execute_code=self.execute_code, work_dir=self.work_dir,
            )
        elif self.topology == "peer_review":
            result = await run_peer_review(
                context, self.model_client, effective_system, p.critic_message,
                execute_code=self.execute_code, work_dir=self.work_dir,
                critic_model_client=self.critic_model_client,
                **iter_kw,
            )
        elif self.topology == "adversarial":
            result = await run_adversarial(
                context, self.model_client, effective_system, p.adversary_message,
                execute_code=self.execute_code, work_dir=self.work_dir,
                adversary_model_client=self.adversary_model_client,
                **iter_kw,
            )
        elif self.topology == "parallel_ensemble":
            result = await run_parallel_ensemble(
                context,
                self.model_client,
                effective_system,
                judge_message=with_final_answer_instruction(p.judge_message, p.final_answer_hint),
                judge_model_client=self.judge_model_client,
                n_experts=self.n_experts,
                execute_code=self.execute_code,
                work_dir=self.work_dir,
                judge_task_template=p.judge_task_template,
                expert_model_clients=self.expert_model_clients,
            )
        elif self.topology == "self_reflection":
            result = await run_self_reflection(
                context, self.model_client, effective_system, p.reflection_prompt,
                execute_code=self.execute_code, work_dir=self.work_dir,
                revision_prompt=p.revision_prompt,
                **iter_kw,
            )
        elif self.topology == "debate":
            result = await run_debate(
                context, self.model_client, effective_system, p.debate_prompt,
                n_agents=self.n_experts,
                agent_model_clients=self.expert_model_clients,
                aggregation=self.aggregation,
                judge_message=with_final_answer_instruction(p.judge_message, p.final_answer_hint),
                judge_model_client=self.judge_model_client,
                judge_task_template=p.judge_task_template,
                answer_extractor=self.answer_extractor,
                answer_normalizer=self.answer_normalizer,
                stop_on_consensus=self.stop_on_consensus,
                execute_code=self.execute_code,
                work_dir=self.work_dir,
                **({} if self.max_rounds is None else {"max_rounds": self.max_rounds}),
            )
        else:
            raise ValueError(f"Unknown topology: {self.topology!r}")

        # One extractor for every topology, applied to the designated final message
        result.final_answer, result.final_answer_tagged = self.answer_extractor(result.response_text)
        if not result.final_answer_tagged:
            print("[TopologyAgent] Warning: no 'FINAL ANSWER:' tag in the final message; "
                  "using the whole message as the final answer.")
        # Token usage of this run (overall and per role), for compute-matched comparisons
        result.usage = summarize_usage(result.messages)

        log_session(self._log_path, self.topology, result, metadata)
        return result


__all__ = ["TopologyAgent", "VALID_TOPOLOGIES"]

