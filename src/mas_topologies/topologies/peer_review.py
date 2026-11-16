"""Peer-review topology: Architect proposes, Reviewer checks. Default max 4 rounds."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ..models import TopologyResult
from ..prompts import CRITIC_SYSTEM_MESSAGE as _DEFAULT_CRITIC
from ..protocol import VERDICT_INSTRUCTION, approval_termination, with_verdict_instruction
from ._executor import make_code_executor_agent
from ._messages import collect_messages


async def run(
    context: str,
    model_client,
    system_message: str,
    critic_message: str = _DEFAULT_CRITIC,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
    max_rounds: int = 4,
    verdict_instruction: str = VERDICT_INSTRUCTION,
    critic_model_client=None,
) -> TopologyResult:
    """Architect proposes, Reviewer checks correctness and design quality.

    Terminates when the Reviewer's own message ends with ``VERDICT: APPROVE``
    (see :mod:`mas_topologies.protocol`) or after the Architect has spoken
    *max_rounds* times.  *verdict_instruction* is appended to the Reviewer's
    prompt unless it already asks for a verdict.  When *execute_code* is
    ``True`` a :class:`CodeExecutorAgent` is inserted between Architect and
    Reviewer so the Reviewer sees real execution output before critiquing.

    *critic_model_client* drives the Reviewer; it falls back to *model_client*
    (which always drives the Architect) when ``None``.
    """
    architect = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )
    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=critic_model_client or model_client,
        system_message=with_verdict_instruction(critic_message, verdict_instruction),
    )

    if execute_code:
        # Only the Architect's code is executed, never code the Reviewer quotes
        agents = [architect, make_code_executor_agent(work_dir, sources=["Architect"]), reviewer]
    else:
        agents = [architect, reviewer]
    # The task message counts towards MaxMessageTermination, so this cap lets
    # the Architect speak exactly max_rounds times.
    max_msgs = len(agents) * max_rounds

    termination = approval_termination("Reviewer") | MaxMessageTermination(max_msgs)
    team = RoundRobinGroupChat(agents, termination_condition=termination)
    result = await team.run(task=context)

    round_num = 0
    messages = collect_messages(result.messages)
    for m in messages:
        if m["source"] == "Architect":
            round_num += 1
        print(f"\n--- {m['source']} (round {round_num}) ---")
        print(m["content"])

    # The final answer is the Architect's last word (its latest revision)
    arch_msgs = [m for m in messages if m["source"] == "Architect"]
    final_text = arch_msgs[-1]["content"] if arch_msgs else ""

    return TopologyResult(
        response_text=final_text,
        messages=messages,
        rounds_used=round_num,
    )
