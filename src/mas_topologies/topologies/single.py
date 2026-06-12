"""Single-agent topology: one LLM call, returns response text."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from ..models import TopologyResult
from ._executor import make_code_executor_agent


async def run(
    context: str,
    model_client,
    system_message: str,
    interactive: bool = False,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
) -> TopologyResult:
    """Single agent, one shot (or interactive console mode).

    When *execute_code* is ``True`` a :class:`CodeExecutorAgent` is added to the
    group chat so the agent can write and run Python code, see the output, and
    refine its answer.  The conversation terminates when the agent says ``DONE``
    or after 10 messages.

    When *execute_code* is ``False`` (default) the existing fast single-call
    path is used unchanged.
    """
    agent = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )

    if execute_code:
        agents = [agent, make_code_executor_agent(work_dir)]
        termination = TextMentionTermination("DONE") | MaxMessageTermination(10)
        team = RoundRobinGroupChat(agents, termination_condition=termination)
        if interactive:
            result = await Console(team.run_stream(task=context))
        else:
            result = await team.run(task=context)
        arch_msgs = [m for m in result.messages if m.source == "Architect"]
        text = arch_msgs[-1].content if arch_msgs else result.messages[-1].content
        messages = [{"source": m.source, "content": m.content} for m in result.messages]
        rounds_used = len(arch_msgs)
    elif interactive:
        termination = TextMentionTermination("APPROVED")
        team = RoundRobinGroupChat([agent], termination_condition=termination)
        result = await Console(team.run_stream(task=context))
        arch_msgs = [m for m in result.messages if m.source == "Architect"]
        text = arch_msgs[-1].content if arch_msgs else result.messages[-1].content
        messages = [{"source": "Architect", "content": text}]
        rounds_used = 1
    else:
        response = await agent.on_messages(
            [TextMessage(content=context, source="user")],
            cancellation_token=None,
        )
        text = response.chat_message.content
        print(text)
        messages = [{"source": "Architect", "content": text}]
        rounds_used = 1

    return TopologyResult(
        response_text=text,
        messages=messages,
        rounds_used=rounds_used,
    )
