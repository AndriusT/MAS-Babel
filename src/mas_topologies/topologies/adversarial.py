"""Adversarial topology: Architect proposes, Adversary challenges. Default max 4 rounds."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ..models import TopologyResult
from ..prompts import ADVERSARIAL_CRITIC_SYSTEM_MESSAGE as _DEFAULT_ADVERSARY
from ..protocol import VERDICT_INSTRUCTION, approval_termination, with_verdict_instruction
from ._executor import make_code_executor_agent
from ._messages import collect_messages


async def run(
    context: str,
    model_client,
    system_message: str,
    adversary_message: str = _DEFAULT_ADVERSARY,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
    max_rounds: int = 4,
    verdict_instruction: str = VERDICT_INSTRUCTION,
    adversary_model_client=None,
) -> TopologyResult:
    """Architect proposes, Adversary challenges.

    Terminates when the Adversary's own message ends with ``VERDICT: APPROVE``
    (it has no substantive objections left) or after the Architect has spoken
    *max_rounds* times.  *verdict_instruction* is appended to the Adversary's
    prompt unless it already asks for a verdict.  When *execute_code* is
    ``True`` a :class:`CodeExecutorAgent` is inserted between Architect and
    Adversary so the Adversary can challenge based on real execution output.

    *adversary_model_client* drives the Adversary; it falls back to
    *model_client* (which always drives the Architect) when ``None``.
    """
    architect = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )
    adversary = AssistantAgent(
        name="Adversary",
        model_client=adversary_model_client or model_client,
        system_message=with_verdict_instruction(adversary_message, verdict_instruction),
    )

    if execute_code:
        # Only the Architect's code is executed, never code the Adversary quotes
        agents = [architect, make_code_executor_agent(work_dir, sources=["Architect"]), adversary]
    else:
        agents = [architect, adversary]
    # The task message counts towards MaxMessageTermination, so this cap lets
    # the Architect speak exactly max_rounds times.
    max_msgs = len(agents) * max_rounds

    termination = approval_termination("Adversary") | MaxMessageTermination(max_msgs)
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
