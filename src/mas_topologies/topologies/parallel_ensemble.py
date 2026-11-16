"""Parallel-ensemble topology: N Architects in parallel, Judge synthesizes."""

import asyncio
import re
from typing import Any, Dict, List, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from ..models import TopologyResult
from ..prompts import JUDGE_SYSTEM_MESSAGE as _DEFAULT_JUDGE
from ..prompts import JUDGE_TASK_TEMPLATE as _DEFAULT_JUDGE_TASK
from ._executor import DoneTermination, make_code_executor_agent
from ._messages import collect_messages, message_dict


_PLACEHOLDER = re.compile(r"\{(context|n|proposals)\}")


def _prepare_judge_input(
    context: str, proposals: list, template: str = _DEFAULT_JUDGE_TASK
) -> str:
    """Build the Judge's task by filling *template* (see ``PromptSet.judge_task_template``).

    Placeholders are filled in a single pass, so braces or placeholder-like text
    inside the context or the proposals (e.g. JSON) are left untouched.
    """
    proposals_text = "".join(
        f"### PROPOSAL {i}:\n{proposal}\n\n" for i, proposal in enumerate(proposals, 1)
    )
    values = {"context": context, "n": str(len(proposals)), "proposals": proposals_text}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


async def _run_one_expert(
    i: int,
    context: str,
    model_client,
    system_message: str,
    execute_code: bool,
    work_dir: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run a single expert, optionally with code execution.

    Returns ``(proposal_text, transcript)`` where *transcript* holds every
    message the expert produced (including CodeRunner output).
    """
    name = f"Architect_{i + 1}"
    arch = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=system_message,
    )
    if execute_code:
        # Each expert gets its own subdirectory to avoid concurrent write collisions
        expert_work_dir = f"{work_dir}/expert_{i + 1}"
        code_runner = make_code_executor_agent(expert_work_dir, sources=[name])
        termination = DoneTermination(name) | MaxMessageTermination(8)
        team = RoundRobinGroupChat([arch, code_runner], termination_condition=termination)
        result = await team.run(task=context)
        transcript = collect_messages(result.messages)
        # Every expert's executor is named "CodeRunner"; tag it with the expert number
        for m in transcript:
            if m["source"] == "CodeRunner":
                m["source"] = f"CodeRunner_{i + 1}"
        arch_msgs = [m for m in transcript if m["source"] == name]
        text = arch_msgs[-1]["content"] if arch_msgs else transcript[-1]["content"]
        return text, transcript
    else:
        response = await arch.on_messages(
            [TextMessage(content=context, source="user")],
            cancellation_token=None,
        )
        text = response.chat_message.content
        return text, [message_dict(response.chat_message)]


async def run(
    context: str,
    model_client,
    system_message: str,
    judge_message: str = _DEFAULT_JUDGE,
    judge_model_client=None,
    n_experts: int = 3,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
    judge_task_template: str = _DEFAULT_JUDGE_TASK,
    expert_model_clients=None,
) -> TopologyResult:
    """Run *n_experts* Architects in parallel, then have the Judge synthesize.

    *judge_model_client* falls back to *model_client* when ``None``.  The Judge's
    task is *judge_task_template* filled with the context and all proposals.
    With *expert_model_clients* (a list), expert *i* uses
    ``expert_model_clients[i % len(expert_model_clients)]`` (mixed-model ensembles);
    otherwise every expert uses *model_client*.

    When *execute_code* is ``True`` each expert runs in a
    :class:`RoundRobinGroupChat` with its own :class:`CodeExecutorAgent` in a
    separate subdirectory (``<work_dir>/expert_N``).  The Judge always operates
    as a pure LLM — it synthesises from proposals that already contain real
    execution results.
    """
    if judge_model_client is None:
        judge_model_client = model_client

    print(f"Running {n_experts} Architects in parallel (execute_code={execute_code})...")

    clients = expert_model_clients or [model_client]
    expert_runs = await asyncio.gather(
        *[
            _run_one_expert(i, context, clients[i % len(clients)], system_message, execute_code, work_dir)
            for i in range(n_experts)
        ]
    )
    proposals = [text for text, _ in expert_runs]

    messages = []
    for i, (text, transcript) in enumerate(expert_runs):
        print(f"\n--- Architect {i + 1} Proposal ---")
        print(text)
        messages.extend(transcript)

    judge = AssistantAgent(
        name="Judge",
        model_client=judge_model_client,
        system_message=judge_message,
    )
    judge_response = await judge.on_messages(
        [TextMessage(
            content=_prepare_judge_input(context, list(proposals), judge_task_template),
            source="user",
        )],
        cancellation_token=None,
    )
    judge_text = judge_response.chat_message.content
    print("\n--- Judge (Synthesis) ---")
    print(judge_text)
    messages.append(message_dict(judge_response.chat_message))

    # The Judge's synthesis is the answer; the proposals stay in *messages*
    return TopologyResult(
        response_text=judge_text,
        messages=messages,
        rounds_used=1,
    )

