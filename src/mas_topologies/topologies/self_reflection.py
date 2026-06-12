"""Self-reflection topology: single agent proposes then self-critiques. Max 3 rounds."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from ..models import TopologyResult
from ..prompts import REFLECTION_PROMPT as _DEFAULT_REFLECTION
from ._executor import contains_code_block, make_code_executor_agent


async def run(
    context: str,
    model_client,
    system_message: str,
    reflection_prompt: str = _DEFAULT_REFLECTION,
    max_rounds: int = 3,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
) -> TopologyResult:
    """Single agent proposes, reflects, and revises up to *max_rounds* times.

    Terminates early when the agent responds with ``REFLECTION_COMPLETE``.

    When *execute_code* is ``True`` any code block in the agent's proposal is
    executed by a :class:`CodeExecutorAgent` before the reflection step, so the
    agent reflects on real output rather than hypothetical output.
    """
    agent = AssistantAgent(
        name="Architect",
        model_client=model_client,
        system_message=system_message,
    )
    code_runner = make_code_executor_agent(work_dir) if execute_code else None

    # Initial proposal
    response = await agent.on_messages(
        [TextMessage(content=context, source="user")],
        cancellation_token=None,
    )
    proposal = response.chat_message.content
    print("\n--- Architect (round 1 - initial) ---")
    print(proposal)

    # Optionally execute any code in the initial proposal
    if code_runner and contains_code_block(proposal):
        exec_resp = await code_runner.on_messages(
            [TextMessage(content=proposal, source="Architect")],
            cancellation_token=None,
        )
        exec_output = exec_resp.chat_message.content
        print("\n--- CodeRunner (round 1) ---")
        print(exec_output)
        # Feed execution output back so the agent reflects on real results
        await agent.on_messages(
            [TextMessage(content=f"Execution output:\n{exec_output}", source="CodeRunner")],
            cancellation_token=None,
        )

    rounds_used = 1
    messages = [{"source": "Architect", "content": proposal}]

    for round_num in range(1, max_rounds + 1):
        # Reflection step
        reflection_response = await agent.on_messages(
            [TextMessage(content=reflection_prompt, source="user")],
            cancellation_token=None,
        )
        reflection = reflection_response.chat_message.content
        print(f"\n--- Self-Reflection (round {round_num}) ---")
        print(reflection)
        messages.append({"source": "Reflection", "content": reflection})

        if "REFLECTION_COMPLETE" in reflection:
            break

        # Revision step
        revision_response = await agent.on_messages(
            [
                TextMessage(
                    content=(
                        "Based on your reflection above, provide your COMPLETE revised response "
                        "(full Analysis Section, Python code block, and JSON hyperparameters block)."
                    ),
                    source="user",
                )
            ],
            cancellation_token=None,
        )
        proposal = revision_response.chat_message.content
        rounds_used += 1
        print(f"\n--- Architect (round {round_num + 1} - revised) ---")
        print(proposal)
        messages.append({"source": "Architect", "content": proposal})

        # Execute revised code if enabled
        if code_runner and contains_code_block(proposal):
            exec_resp = await code_runner.on_messages(
                [TextMessage(content=proposal, source="Architect")],
                cancellation_token=None,
            )
            exec_output = exec_resp.chat_message.content
            print(f"\n--- CodeRunner (round {round_num + 1}) ---")
            print(exec_output)
            messages.append({"source": "CodeRunner", "content": exec_output})
            await agent.on_messages(
                [TextMessage(content=f"Execution output:\n{exec_output}", source="CodeRunner")],
                cancellation_token=None,
            )

    return TopologyResult(
        response_text=proposal,
        messages=messages,
        rounds_used=rounds_used,
    )

