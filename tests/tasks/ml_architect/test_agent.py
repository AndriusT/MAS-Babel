"""Tests for DesignerPlannerAgent (ML-task wrapper around TopologyAgent)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mas_topologies.agent import VALID_TOPOLOGIES
from mas_topologies.models import TopologyResult
from mas_topologies.tasks.ml_architect import DesignerPlannerAgent, DesignInput, DesignOutput
from mas_topologies.tasks.ml_architect.prompts import ML_PROMPT_SET

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DESIGN_INPUT = DesignInput(
    dataset_name="MNIST",
    dataset_description="Handwritten digits",
    num_classes=10,
    input_shape=(28, 28, 1),
    dataset_size=60000,
)

_JSON_BLOCK = (
    '{"hyperparameters": {"optimizer": "adam", "learning_rate": 0.001,'
    ' "batch_size": 64, "epochs": 10, "weight_decay": 0.0,'
    ' "learning_rate_schedule": "none", "warmup_epochs": 0, "dropout_rate": 0.25}}'
)
MOCK_RESULT = TopologyResult(
    response_text=f"Analysis.\n```json\n{_JSON_BLOCK}\n```\n",
    messages=[{"source": "Architect", "content": "analysis"}],
    rounds_used=1,
)


def _agent(topology="single", **kwargs) -> DesignerPlannerAgent:
    return DesignerPlannerAgent(model_client=object(), topology=topology, **kwargs)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            DesignerPlannerAgent(model_client=object(), topology="bad")

    def test_all_valid_topologies_accepted(self):
        for t in VALID_TOPOLOGIES:
            agent = DesignerPlannerAgent(model_client=object(), topology=t)
            assert agent.topology == t

    def test_log_path_none_without_log_dir(self):
        assert _agent()._log_path is None

    def test_log_path_set_when_log_dir_provided(self, tmp_path):
        agent = DesignerPlannerAgent(model_client=object(), log_dir=str(tmp_path))
        assert agent._log_path is not None
        assert agent._log_path.endswith(".jsonl")

    def test_uses_ml_prompt_set_by_default(self):
        agent = _agent()
        assert agent._agent.prompts is ML_PROMPT_SET

    def test_custom_system_message_overrides_default(self):
        agent = DesignerPlannerAgent(
            model_client=object(), system_message="custom prompt"
        )
        assert agent._agent.prompts.system_message == "custom prompt"

    def test_custom_system_message_keeps_ml_critic(self):
        agent = DesignerPlannerAgent(
            model_client=object(), system_message="custom"
        )
        assert agent._agent.prompts.critic_message == ML_PROMPT_SET.critic_message


# ---------------------------------------------------------------------------
# design_architecture delegation
# ---------------------------------------------------------------------------

class TestDesignArchitecture:
    async def _run(self, topology="single", **kwargs):
        agent = _agent(topology=topology, **kwargs)
        with patch.object(agent._agent, "run", new_callable=AsyncMock, return_value=MOCK_RESULT) as m:
            result = await agent.design_architecture(DESIGN_INPUT)
        return result, m

    async def test_returns_design_output(self):
        result, _ = await self._run()
        assert isinstance(result, DesignOutput)

    async def test_parsed_hyperparameters_correct(self):
        result, _ = await self._run()
        assert result.hyperparameters["optimizer"] == "adam"
        assert result.hyperparameters["learning_rate"] == 0.001

    async def test_inner_agent_run_called_once(self):
        _, m = await self._run()
        m.assert_awaited_once()

    async def test_context_string_contains_dataset_name(self):
        agent = _agent()
        with patch.object(agent._agent, "run", new_callable=AsyncMock, return_value=MOCK_RESULT) as m:
            await agent.design_architecture(DESIGN_INPUT)
        context_arg = m.call_args.args[0]
        assert "MNIST" in context_arg

    async def test_metadata_contains_dataset_fields(self):
        agent = _agent()
        with patch.object(agent._agent, "run", new_callable=AsyncMock, return_value=MOCK_RESULT) as m:
            await agent.design_architecture(DESIGN_INPUT)
        metadata = m.call_args.kwargs.get("metadata") or m.call_args.args[2]
        assert metadata["dataset"] == "MNIST"
        assert metadata["iteration"] == 1
        assert metadata["input_shape"] == [28, 28, 1]
        assert metadata["dataset_size"] == 60000

    async def test_interactive_flag_forwarded(self):
        agent = _agent()
        with patch.object(agent._agent, "run", new_callable=AsyncMock, return_value=MOCK_RESULT) as m:
            await agent.design_architecture(DESIGN_INPUT, interactive=True)
        assert m.call_args.args[1] is True
