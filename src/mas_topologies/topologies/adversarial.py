"""Adversarial topology: Architect proposes, Adversary challenges. Max 4 rounds."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ..models import TopologyResult
from ..prompts import ADVERSARIAL_CRITIC_SYSTEM_MESSAGE as _DEFAULT_ADVERSARY
from ._executor import make_code_executor_agent


async def run(
    context: str,
    model_client,
    system_message: str,
    adversary_message: str = _DEFAULT_ADVERSARY,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
) -> TopologyResult:
    """Architect proposes, Adversary challenges. Max 4 rounds.

    Terminates when Adversary responds with ``ACCEPTED``
    or after 8 messages (4 full rounds).  When *execute_code* is ``True`` a
    :class:`CodeExecutorAgent` is inserted between Architect and Adversary so
    the Adversary can challenge the architect based on real execution output.
    MaxMessages is raised to 12 to account for the extra CodeRunner turns.
    """
    architect = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )
    adversary = AssistantAgent(
        name="Adversary",
        model_client=model_client,
        system_message=adversary_message,
    )

    if execute_code:
        agents = [architect, make_code_executor_agent(work_dir), adversary]
        max_msgs = 12   # 3 agents × 4 rounds
    else:
        agents = [architect, adversary]
        max_msgs = 8    # 2 agents × 4 rounds

    termination = TextMentionTermination("ACCEPTED") | MaxMessageTermination(max_msgs)
    team = RoundRobinGroupChat(agents, termination_condition=termination)
    result = await team.run(task=context)

    round_num = 0
    messages = []
    for msg in result.messages:
        if msg.source == "Architect":
            round_num += 1
            print(f"\n--- Architect (round {round_num}) ---")
            print(msg.content)
            messages.append({"source": "Architect", "content": msg.content})
        elif msg.source == "Adversary":
            print(f"\n--- Adversary (round {round_num}) ---")
            print(msg.content)
            messages.append({"source": "Adversary", "content": msg.content})

    arch_msgs = [m for m in messages if m["source"] == "Architect"]
    final_text = None
    for m in reversed(arch_msgs):
        if "```json" in m["content"]:
            final_text = m["content"]
            break
    if final_text is None:
        final_text = arch_msgs[-1]["content"] if arch_msgs else messages[-1]["content"]

    return TopologyResult(
        response_text=final_text,
        messages=messages,
        rounds_used=round_num,
    )
