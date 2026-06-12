"""Tests for TopologyAgent dispatch logic."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from mas_topologies.agent import TopologyAgent, VALID_TOPOLOGIES
from mas_topologies.models import TopologyResult
from mas_topologies.prompts import PromptSet

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_PROMPT_SET = PromptSet(
    system_message="You are a test agent.",
    critic_message="Review critically.",
    adversary_message="Challenge everything.",
    judge_message="Synthesise the best.",
    reflection_prompt="Reflect on your output.",
)

CONTEXT = "A plain context string for the task."

MOCK_RESULT = TopologyResult(
    response_text="Test response.",
    messages=[{"source": "Architect", "content": "response"}],
    rounds_used=1,
)


def _agent(topology="single", **kwargs) -> TopologyAgent:
    return TopologyAgent(
        model_client=object(), topology=topology, prompts=MOCK_PROMPT_SET, **kwargs
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_unknown_topology_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            TopologyAgent(model_client=object(), topology="nonexistent", prompts=MOCK_PROMPT_SET)

    def test_no_prompts_raises_value_error(self):
        with pytest.raises(ValueError, match="PromptSet"):
            TopologyAgent(model_client=object())

    def test_all_valid_topologies_accepted(self):
        for t in VALID_TOPOLOGIES:
            agent = TopologyAgent(model_client=object(), topology=t, prompts=MOCK_PROMPT_SET)
            assert agent.topology == t

    def test_log_path_set_when_log_dir_provided(self, tmp_path):
        agent = TopologyAgent(
            model_client=object(), prompts=MOCK_PROMPT_SET, log_dir=str(tmp_path)
        )
        assert agent._log_path is not None
        assert str(tmp_path) in agent._log_path
        assert agent._log_path.endswith(".jsonl")

    def test_log_dir_created_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "logs")
        TopologyAgent(model_client=object(), prompts=MOCK_PROMPT_SET, log_dir=new_dir)
        assert os.path.isdir(new_dir)

    def test_no_log_path_without_log_dir(self):
        assert _agent()._log_path is None

    def test_prompts_stored(self):
        agent = _agent()
        assert agent.prompts is MOCK_PROMPT_SET

    def test_judge_model_client_defaults_to_model_client(self):
        mc = object()
        agent = TopologyAgent(model_client=mc, prompts=MOCK_PROMPT_SET)
        assert agent.judge_model_client is mc

    def test_judge_model_client_stored_when_provided(self):
        mc1, mc2 = object(), object()
        agent = TopologyAgent(model_client=mc1, prompts=MOCK_PROMPT_SET, judge_model_client=mc2)
        assert agent.judge_model_client is mc2

    def test_n_experts_stored(self):
        agent = TopologyAgent(model_client=object(), prompts=MOCK_PROMPT_SET, n_experts=5)
        assert agent.n_experts == 5


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------

class TestDispatch:
    """Verify run() routes to the correct topology function."""

    async def _dispatch(self, topology, mock_name, **agent_kwargs):
        agent = _agent(topology=topology, **agent_kwargs)
        with patch(
            f"mas_topologies.agent.{mock_name}",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            result = await agent.run(CONTEXT)
        return result, m

    async def test_single_calls_run_single(self):
        _, m = await self._dispatch("single", "run_single")
        m.assert_awaited_once()

    async def test_peer_review_calls_run_peer_review(self):
        _, m = await self._dispatch("peer_review", "run_peer_review")
        m.assert_awaited_once()

    async def test_adversarial_calls_run_adversarial(self):
        _, m = await self._dispatch("adversarial", "run_adversarial")
        m.assert_awaited_once()

    async def test_self_reflection_calls_run_self_reflection(self):
        _, m = await self._dispatch("self_reflection", "run_self_reflection")
        m.assert_awaited_once()

    async def test_parallel_ensemble_calls_run_parallel_ensemble(self):
        _, m = await self._dispatch(
            "parallel_ensemble", "run_parallel_ensemble", n_experts=2
        )
        m.assert_awaited_once()

    async def test_returns_topology_result(self):
        result, _ = await self._dispatch("single", "run_single")
        assert isinstance(result, TopologyResult)
        assert result.response_text == MOCK_RESULT.response_text

    async def test_context_string_passed_as_first_arg(self):
        agent = _agent(topology="single")
        with patch(
            "mas_topologies.agent.run_single",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.args[0] == CONTEXT

    async def test_system_message_from_prompts_forwarded_to_single(self):
        agent = _agent(topology="single")
        with patch(
            "mas_topologies.agent.run_single",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.args[2] == MOCK_PROMPT_SET.system_message

    async def test_critic_message_forwarded_to_peer_review(self):
        agent = _agent(topology="peer_review")
        with patch(
            "mas_topologies.agent.run_peer_review",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.args[3] == MOCK_PROMPT_SET.critic_message

    async def test_adversary_message_forwarded_to_adversarial(self):
        agent = _agent(topology="adversarial")
        with patch(
            "mas_topologies.agent.run_adversarial",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.args[3] == MOCK_PROMPT_SET.adversary_message

    async def test_reflection_prompt_forwarded_to_self_reflection(self):
        agent = _agent(topology="self_reflection")
        with patch(
            "mas_topologies.agent.run_self_reflection",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.args[3] == MOCK_PROMPT_SET.reflection_prompt

    async def test_n_experts_forwarded_to_parallel_ensemble(self):
        agent = _agent(topology="parallel_ensemble", n_experts=7)
        with patch(
            "mas_topologies.agent.run_parallel_ensemble",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT)
        assert m.call_args.kwargs["n_experts"] == 7

    async def test_interactive_flag_forwarded_to_single(self):
        agent = _agent(topology="single")
        with patch(
            "mas_topologies.agent.run_single",
            new_callable=AsyncMock,
            return_value=MOCK_RESULT,
        ) as m:
            await agent.run(CONTEXT, interactive=True)
        assert m.call_args.args[3] is True


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------

class TestLoggingIntegration:
    async def test_log_session_called(self):
        agent = _agent(topology="single")
        with patch("mas_topologies.agent.run_single", new_callable=AsyncMock, return_value=MOCK_RESULT):
            with patch("mas_topologies.agent.log_session") as log_mock:
                await agent.run(CONTEXT)
        log_mock.assert_called_once()

    async def test_log_called_with_none_when_no_log_dir(self):
        agent = _agent(topology="single")
        with patch("mas_topologies.agent.run_single", new_callable=AsyncMock, return_value=MOCK_RESULT):
            with patch("mas_topologies.agent.log_session") as log_mock:
                await agent.run(CONTEXT)
        assert log_mock.call_args.args[0] is None

    async def test_log_called_with_correct_topology(self):
        agent = _agent(topology="peer_review")
        with patch("mas_topologies.agent.run_peer_review", new_callable=AsyncMock, return_value=MOCK_RESULT):
            with patch("mas_topologies.agent.log_session") as log_mock:
                await agent.run(CONTEXT)
        assert log_mock.call_args.args[1] == "peer_review"

    async def test_log_called_with_topology_result(self):
        agent = _agent(topology="single")
        with patch("mas_topologies.agent.run_single", new_callable=AsyncMock, return_value=MOCK_RESULT):
            with patch("mas_topologies.agent.log_session") as log_mock:
                await agent.run(CONTEXT)
        assert log_mock.call_args.args[2] is MOCK_RESULT

    async def test_metadata_forwarded_to_log_session(self):
        agent = _agent(topology="single")
        meta = {"task": "test", "iteration": 1}
        with patch("mas_topologies.agent.run_single", new_callable=AsyncMock, return_value=MOCK_RESULT):
            with patch("mas_topologies.agent.log_session") as log_mock:
                await agent.run(CONTEXT, metadata=meta)
        assert log_mock.call_args.args[3] == meta

    async def test_log_file_written_end_to_end(self, tmp_path):
        """Integration: real log_session writes a JSONL file."""
        agent = TopologyAgent(
            model_client=object(),
            prompts=MOCK_PROMPT_SET,
            topology="single",
            log_dir=str(tmp_path),
        )
        with patch("mas_topologies.agent.run_single", new_callable=AsyncMock, return_value=MOCK_RESULT):
            await agent.run(CONTEXT)
        import json
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        entry = json.loads(jsonl_files[0].read_text())
        assert entry["topology"] == "single"
        assert "timestamp" in entry

