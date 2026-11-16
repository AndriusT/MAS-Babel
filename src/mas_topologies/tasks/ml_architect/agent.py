"""DesignerPlannerAgent — thin ML-task wrapper around TopologyAgent."""

import dataclasses
from typing import Optional

from mas_topologies.agent import TopologyAgent
from mas_topologies.prompts import PromptSet

from .context import parse_response, prepare_context
from .models import DesignInput, DesignOutput
from .prompts import ML_PROMPT_SET


class DesignerPlannerAgent:
    """ML architecture design agent.

    Wraps :class:`~mas_topologies.agent.TopologyAgent` with the ML-specific
    :data:`~mas_topologies.tasks.ml_architect.prompts.ML_PROMPT_SET`.  Converts a
    :class:`DesignInput` to a context string, dispatches to the configured topology,
    and parses the result back to a :class:`DesignOutput`.

    Args:
        model_client: AutoGen-compatible model client.
        topology: One of the supported topology names.
        system_message: Override the default architect system prompt (optional).
        log_dir: Directory for JSONL session logs.  ``None`` disables logging.
        n_experts: Parallel Architect count (``parallel_ensemble`` only).
        judge_model_client: Separate client for the Judge agent.
        max_rounds: Max Architect turns for iterative topologies (``None`` keeps
            each topology's default).
        critic_model_client: Separate client for the Reviewer (``peer_review``).
        adversary_model_client: Separate client for the Adversary (``adversarial``).
        expert_model_clients: List of clients for the parallel experts / debaters
            (agent *i* uses entry ``i % len``); ``None`` uses *model_client* for all.
        aggregation: How ``debate`` picks the final answer.  Defaults to
            ``"judge"`` here: the ML answer is code, where exact-match voting
            is meaningless.
    """

    def __init__(
        self,
        model_client,
        topology: str = "single",
        system_message: Optional[str] = None,
        log_dir: Optional[str] = None,
        n_experts: int = 3,
        judge_model_client=None,
        max_rounds: Optional[int] = None,
        critic_model_client=None,
        adversary_model_client=None,
        expert_model_clients=None,
        aggregation: str = "judge",
    ):
        prompts: PromptSet
        if system_message is None:
            prompts = ML_PROMPT_SET
        else:
            # Keep every other ML field (critic, judge, hints, ...) as is
            prompts = dataclasses.replace(ML_PROMPT_SET, system_message=system_message)

        self._agent = TopologyAgent(
            model_client=model_client,
            topology=topology,
            prompts=prompts,
            log_dir=log_dir,
            n_experts=n_experts,
            judge_model_client=judge_model_client,
            max_rounds=max_rounds,
            critic_model_client=critic_model_client,
            adversary_model_client=adversary_model_client,
            expert_model_clients=expert_model_clients,
            aggregation=aggregation,
        )

    @property
    def topology(self) -> str:
        return self._agent.topology

    @property
    def _log_path(self) -> Optional[str]:
        return self._agent._log_path

    async def design_architecture(
        self,
        design_input: DesignInput,
        interactive: bool = False,
    ) -> DesignOutput:
        """Run the topology and return a structured :class:`DesignOutput`.

        Task-specific metadata (iteration, dataset, input shape) is forwarded
        to the session log via :func:`~mas_topologies.logging.log_session`.
        """
        context = prepare_context(design_input)
        metadata = {
            "iteration": design_input.iteration_number,
            "dataset": design_input.dataset_name,
            "input_shape": list(design_input.input_shape),
            "dataset_size": design_input.dataset_size,
        }
        result = await self._agent.run(context, interactive, metadata=metadata)
        # response_text is exactly the final message (the Judge's for ensembles), so
        # its last JSON block is the final answer's; parsing the whole message also
        # keeps the Analysis in proposed_architecture.  result.final_answer holds the
        # tagged answer itself (code + hyperparameters) for benchmark scoring.
        return parse_response(result.response_text, design_input)


__all__ = ["DesignerPlannerAgent"]
