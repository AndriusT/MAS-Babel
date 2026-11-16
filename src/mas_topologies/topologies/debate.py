"""Debate topology: N debaters answer, read each other's answers and revise; vote or judge.

Round 1: every debater answers the task independently (concurrently).
Rounds 2..max_rounds: every debater receives the OTHER debaters' latest answers
(``PromptSet.debate_prompt``) and gives an updated answer.  Each debater keeps its
own history; only the others' latest answers are passed, so tokens grow linearly
with the number of rounds.

The run stops early when all debaters' normalised final answers agree (consensus).
The final answer is the majority vote over the debaters' last answers
(``aggregation="vote"``), or a Judge's synthesis of them (``"judge"``, for code or
free-form answers where exact voting is meaningless).

With ``max_rounds=1`` and ``aggregation="vote"`` this is self-consistency:
N independent answers plus a majority vote.

With ``execute_code=True`` every debater has its own CodeRunner (own work
directory, executing only that debater's code).  After each round, each answer's
code is run; in the next round a debater sees its own execution output and the
others' answers together with their outputs, and a Judge sees each final
proposal's output.
"""

import asyncio
import re
from typing import Callable, List, Optional, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from ..models import TopologyResult
from ..prompts import DEBATE_PROMPT as _DEFAULT_DEBATE
from ..prompts import JUDGE_SYSTEM_MESSAGE as _DEFAULT_JUDGE
from ..prompts import JUDGE_TASK_TEMPLATE as _DEFAULT_JUDGE_TASK
from ..protocol import extract_final_answer, majority_vote, normalize_answer
from ._executor import contains_code_block, make_code_executor_agent
from ._messages import message_dict
from .parallel_ensemble import _prepare_judge_input

AGGREGATIONS = ("vote", "judge")

_PLACEHOLDER = re.compile(r"\{(others|round)\}")
_MAX_LOGGED_ANSWER = 200   # normalised answers stored in result.info are truncated


def _with_output(text: str, output: Optional[str]) -> str:
    """An answer followed by its code-execution output (when its code was run)."""
    return text if output is None else f"{text}\n\n#### Execution output:\n{output}"


def _others_text(i: int, latest: List[str], outputs: Optional[List[Optional[str]]] = None) -> str:
    """The latest answers (+ execution outputs) of every debater except debater *i*."""
    outputs = outputs or [None] * len(latest)
    return "".join(
        f"### Debater {j + 1}:\n{_with_output(text, outputs[j])}\n\n"
        for j, text in enumerate(latest) if j != i
    )


def _fill_debate_prompt(template: str, others: str, round_num: int) -> str:
    """Fill ``{others}`` / ``{round}`` in one pass (braces inside answers are left alone)."""
    values = {"others": others, "round": str(round_num)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


async def run(
    context: str,
    model_client,
    system_message: str,
    debate_prompt: str = _DEFAULT_DEBATE,
    n_agents: int = 3,
    max_rounds: int = 3,
    agent_model_clients=None,
    aggregation: str = "vote",
    judge_message: str = _DEFAULT_JUDGE,
    judge_model_client=None,
    judge_task_template: str = _DEFAULT_JUDGE_TASK,
    answer_extractor: Callable[[str], Tuple[str, bool]] = extract_final_answer,
    answer_normalizer: Callable[[str], str] = normalize_answer,
    stop_on_consensus: bool = True,
    execute_code: bool = False,
    work_dir: str = "code_workspace",
) -> TopologyResult:
    """Run *n_agents* debaters for up to *max_rounds* rounds, then vote or judge.

    *agent_model_clients* (a list) gives debater *i* the client
    ``agent_model_clients[i % len]``; otherwise all use *model_client*.  The
    Judge (``aggregation="judge"`` only) uses *judge_model_client*, falling back
    to *model_client*.  *answer_extractor* / *answer_normalizer* turn each
    debater's message into a comparable answer for the consensus check and vote.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if n_agents < 1 or max_rounds < 1:
        raise ValueError("n_agents and max_rounds must both be >= 1")

    clients = agent_model_clients or [model_client]
    debaters = [
        AssistantAgent(
            name=f"Debater_{i + 1}",
            model_client=clients[i % len(clients)],
            system_message=system_message,
        )
        for i in range(n_agents)
    ]

    # One CodeRunner per debater, in its own work directory, running only its code
    runners = [
        make_code_executor_agent(f"{work_dir}/debater_{i + 1}", sources=[d.name])
        for i, d in enumerate(debaters)
    ] if execute_code else None

    async def ask(agent: AssistantAgent, content: str, own_output: Optional[str] = None):
        incoming = []
        if own_output is not None:
            incoming.append(TextMessage(
                content=f"Execution output of your previous answer:\n{own_output}", source="CodeRunner"))
        incoming.append(TextMessage(content=content, source="user"))
        response = await agent.on_messages(incoming, cancellation_token=None)
        return response.chat_message

    async def run_code(i: int, reply) -> Optional[str]:
        """Execute debater *i*'s code (None when there is no code block)."""
        text = reply.to_text()
        if runners is None or not contains_code_block(text):
            return None
        result = await runners[i].on_messages(
            [TextMessage(content=text, source=debaters[i].name)],
            cancellation_token=CancellationToken(),   # the executor needs a real token
        )
        return result.chat_message.to_text()

    messages = []
    answers_by_round = []
    consensus_round = None

    async def record(round_num: int, replies) -> Tuple[List[str], List[Optional[str]]]:
        """Log one round (+ run its code); return normalised answers and execution outputs."""
        print(f"\n=== Debate round {round_num} ===")
        for agent, reply in zip(debaters, replies):
            print(f"\n--- {agent.name} (round {round_num}) ---")
            print(reply.to_text())
            messages.append({**message_dict(reply), "round": round_num})
        outputs = list(await asyncio.gather(*[run_code(i, r) for i, r in enumerate(replies)]))
        for i, out in enumerate(outputs):
            if out is not None:
                print(f"\n--- CodeRunner_{i + 1} (round {round_num}) ---")
                print(out)
                messages.append({"source": f"CodeRunner_{i + 1}", "content": out, "round": round_num})
        normalised = [answer_normalizer(answer_extractor(r.to_text())[0]) for r in replies]
        answers_by_round.append([a[:_MAX_LOGGED_ANSWER] for a in normalised])
        return normalised, outputs

    print(f"Running a debate: {n_agents} debaters, up to {max_rounds} round(s), "
          f"aggregation={aggregation}, execute_code={execute_code}")
    latest = await asyncio.gather(*[ask(d, context) for d in debaters])
    rounds_used = 1
    normalised, outputs = await record(1, latest)
    if len(set(normalised)) == 1:
        consensus_round = 1

    for round_num in range(2, max_rounds + 1):
        if stop_on_consensus and consensus_round is not None:
            break
        texts = [r.to_text() for r in latest]
        latest = await asyncio.gather(*[
            ask(d, _fill_debate_prompt(debate_prompt, _others_text(i, texts, outputs), round_num),
                own_output=outputs[i])
            for i, d in enumerate(debaters)
        ])
        rounds_used = round_num
        normalised, outputs = await record(round_num, latest)
        if len(set(normalised)) == 1 and consensus_round is None:
            consensus_round = round_num

    final_texts = [r.to_text() for r in latest]
    info = {
        "aggregation": aggregation,
        "n_agents": n_agents,
        "consensus_round": consensus_round,
        "answers_by_round": answers_by_round,
    }

    if aggregation == "vote":
        answers = [answer_extractor(t)[0] for t in final_texts]
        winner, counts, tie = majority_vote(answers, answer_normalizer)
        # The winner's full message is the final message, so the central extractor
        # in TopologyAgent yields exactly the voted answer
        response_text = final_texts[winner]
        info.update(votes=counts, winner=debaters[winner].name, tie=tie)
        print(f"\n--- Vote: {counts}  winner={debaters[winner].name}{'  (tie)' if tie else ''} ---")
    else:
        judge = AssistantAgent(
            name="Judge",
            model_client=judge_model_client or model_client,
            system_message=judge_message,
        )
        # The Judge sees each final proposal together with its execution output (if run)
        proposals = [_with_output(t, out) for t, out in zip(final_texts, outputs)]
        judge_reply = await ask(judge, _prepare_judge_input(context, proposals, judge_task_template))
        response_text = judge_reply.to_text()
        messages.append({**message_dict(judge_reply), "round": rounds_used})
        print("\n--- Judge (Synthesis) ---")
        print(response_text)

    return TopologyResult(
        response_text=response_text,
        messages=messages,
        rounds_used=rounds_used,
        info=info,
    )
