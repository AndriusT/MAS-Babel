"""Self-reflection topology: single agent proposes then self-critiques. Default max 3 rounds."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from ..models import TopologyResult
from ..prompts import REFLECTION_PROMPT as _DEFAULT_REFLECTION
from ..prompts import REVISION_PROMPT as _DEFAULT_REVISION
from ..protocol import VERDICT_INSTRUCTION, is_approval, with_verdict_instruction
from ._executor import contains_code_block, make_code_executor_agent
from ._messages import message_dict

# Source of the agent's reply to code-execution output (logged, but not a proposal)
EXEC_REPLY_SOURCE = "Architect (on execution output)"


async def run(
    context: str,
    model_client,
    system_message: str,
    reflection_prompt: str = _DEFAULT_REFLECTION,
    max_rounds: int = 3,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
    verdict_instruction: str = VERDICT_INSTRUCTION,
    revision_prompt: str = _DEFAULT_REVISION,
) -> TopologyResult:
    """Single agent proposes, reflects, and revises (*revision_prompt* asks for
    the revised answer after each reflection).

    *max_rounds* is the maximum number of proposals (the initial one plus
    revisions), matching the meaning of *max_rounds* in the other topologies.
    Terminates early when a reflection ends with ``VERDICT: APPROVE``
    (see :mod:`mas_topologies.protocol`).

    When *execute_code* is ``True`` any code block in the agent's proposal is
    executed by a :class:`CodeExecutorAgent` before the reflection step, so the
    agent reflects on real output rather than hypothetical output.
    """
    agent = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )
    code_runner = make_code_executor_agent(work_dir, sources=["Architect"]) if execute_code else None
    reflection_prompt = with_verdict_instruction(reflection_prompt, verdict_instruction)

    # Initial proposal
    response = await agent.on_messages(
        [TextMessage(content=context, source="user")],
        cancellation_token=None,
    )
    proposal = response.chat_message.content
    print("\n--- Architect (round 1 - initial) ---")
    print(proposal)
    messages = [message_dict(response.chat_message)]

    # Optionally execute any code in the initial proposal
    if code_runner and contains_code_block(proposal):
        # The code executor links the subprocess to the token, so it cannot be None
        exec_resp = await code_runner.on_messages(
            [TextMessage(content=proposal, source="Architect")],
            cancellation_token=CancellationToken(),
        )
        exec_output = exec_resp.chat_message.content
        print("\n--- CodeRunner (round 1) ---")
        print(exec_output)
        messages.append({"source": "CodeRunner", "content": exec_output})
        # Feed execution output back so the agent reflects on real results.  Its reply
        # is a real (paid) model call, so log it under its own source: it is neither a
        # new proposal nor the final answer.
        feedback = await agent.on_messages(
            [TextMessage(content=f"Execution output:\n{exec_output}", source="CodeRunner")],
            cancellation_token=None,
        )
        messages.append(message_dict(feedback.chat_message, source=EXEC_REPLY_SOURCE))

    rounds_used = 1

    # Each iteration adds one revised proposal, so at most max_rounds proposals in total
    for round_num in range(1, max_rounds):
        # Reflection step
        reflection_response = await agent.on_messages(
            [TextMessage(content=reflection_prompt, source="user")],
            cancellation_token=None,
        )
        reflection = reflection_response.chat_message.content
        print(f"\n--- Self-Reflection (round {round_num}) ---")
        print(reflection)
        messages.append(message_dict(reflection_response.chat_message, source="Reflection"))

        if is_approval(reflection):
            break

        # Revision step
        revision_response = await agent.on_messages(
            [TextMessage(content=revision_prompt, source="user")],
            cancellation_token=None,
        )
        proposal = revision_response.chat_message.content
        rounds_used += 1
        print(f"\n--- Architect (round {round_num + 1} - revised) ---")
        print(proposal)
        messages.append(message_dict(revision_response.chat_message))

        # Execute revised code if enabled
        if code_runner and contains_code_block(proposal):
            exec_resp = await code_runner.on_messages(
                [TextMessage(content=proposal, source="Architect")],
                cancellation_token=CancellationToken(),
            )
            exec_output = exec_resp.chat_message.content
            print(f"\n--- CodeRunner (round {round_num + 1}) ---")
            print(exec_output)
            messages.append({"source": "CodeRunner", "content": exec_output})
            feedback = await agent.on_messages(
                [TextMessage(content=f"Execution output:\n{exec_output}", source="CodeRunner")],
                cancellation_token=None,
            )
            messages.append(message_dict(feedback.chat_message, source=EXEC_REPLY_SOURCE))

    return TopologyResult(
        response_text=proposal,
        messages=messages,
        rounds_used=rounds_used,
    )

