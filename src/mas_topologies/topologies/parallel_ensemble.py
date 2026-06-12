"""Parallel-ensemble topology: N Architects in parallel, Judge synthesizes."""

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from ..models import TopologyResult
from ..prompts import JUDGE_SYSTEM_MESSAGE as _DEFAULT_JUDGE
from ._executor import make_code_executor_agent


def _prepare_judge_input(context: str, proposals: list) -> str:
    """Build the Judge's task from the original context and all proposals."""
    n = len(proposals)
    judge_input = f"""## ORIGINAL TASK
{context}

## PROPOSALS FROM {n} ARCHITECTS

"""
    for i, proposal in enumerate(proposals, 1):
        judge_input += f"### ARCHITECT {i} PROPOSAL:\n{proposal}\n\n"

    judge_input += f"""## YOUR TASK
Analyze these {n} proposals. Compare them on code correctness, architecture trade-offs,
capacity fit, regularization, hyperparameters, and task alignment.

Then synthesize ONE final proposal by either:
- Selecting the best single proposal, OR
- Combining the strongest ideas from multiple proposals

Your output MUST be a complete, synthesized architecture with:
1. Comparative Analysis section
2. Complete Python code block with create_model(input_shape, num_classes)
3. JSON hyperparameters block

Output ONLY the final proposal — no explanations outside the three sections above.
"""
    return judge_input


async def _run_one_expert(
    i: int,
    context: str,
    model_client,
    system_message: str,
    execute_code: bool,
    work_dir: str,
) -> str:
    """Run a single expert, optionally with code execution.  Returns proposal text."""
    arch = AssistantAgent(
        name=f"Architect_{i + 1}",
        model_client=model_client,
        system_message=system_message,
    )
    if execute_code:
        # Each expert gets its own subdirectory to avoid concurrent write collisions
        expert_work_dir = f"{work_dir}/expert_{i + 1}"
        code_runner = make_code_executor_agent(expert_work_dir)
        termination = TextMentionTermination("DONE") | MaxMessageTermination(8)
        team = RoundRobinGroupChat([arch, code_runner], termination_condition=termination)
        result = await team.run(task=context)
        arch_msgs = [m for m in result.messages if m.source == f"Architect_{i + 1}"]
        return arch_msgs[-1].content if arch_msgs else result.messages[-1].content
    else:
        response = await arch.on_messages(
            [TextMessage(content=context, source="user")],
            cancellation_token=None,
        )
        return response.chat_message.content


async def run(
    context: str,
    model_client,
    system_message: str,
    judge_message: str = _DEFAULT_JUDGE,
    judge_model_client=None,
    n_experts: int = 3,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
) -> TopologyResult:
    """Run *n_experts* Architects in parallel, then have the Judge synthesize.

    *judge_model_client* falls back to *model_client* when ``None``.

    When *execute_code* is ``True`` each expert runs in a
    :class:`RoundRobinGroupChat` with its own :class:`CodeExecutorAgent` in a
    separate subdirectory (``<work_dir>/expert_N``).  The Judge always operates
    as a pure LLM — it synthesises from proposals that already contain real
    execution results.
    """
    if judge_model_client is None:
        judge_model_client = model_client

    print(f"Running {n_experts} Architects in parallel (execute_code={execute_code})...")

    proposals = await asyncio.gather(
        *[
            _run_one_expert(i, context, model_client, system_message, execute_code, work_dir)
            for i in range(n_experts)
        ]
    )

    messages = []
    for i, text in enumerate(proposals):
        print(f"\n--- Architect {i + 1} Proposal ---")
        print(text)
        messages.append({"source": f"Architect_{i + 1}", "content": text})

    judge = AssistantAgent(
        name="Judge",
        model_client=judge_model_client,
        system_message=judge_message,
    )
    judge_response = await judge.on_messages(
        [TextMessage(content=_prepare_judge_input(context, list(proposals)), source="user")],
        cancellation_token=None,
    )
    judge_text = judge_response.chat_message.content
    print("\n--- Judge (Synthesis) ---")
    print(judge_text)
    messages.append({"source": "Judge", "content": judge_text})

    full_session = ""
    for i, p in enumerate(proposals, 1):
        full_session += f"\n--- Architect {i} Proposal ---\n{p}\n"
    full_session += f"\n--- Judge (Synthesis) ---\n{judge_text}"

    return TopologyResult(
        response_text=full_session,
        messages=messages,
        rounds_used=1,
    )

