"""PromptSet contract and domain-neutral default prompts.

The :class:`PromptSet` dataclass is the generic contract between any task and the
topology machinery.  Every role prompt has a domain-neutral default defined
below, so a task only needs to supply what it wants to specialise.  Task-specific
prompt sets (e.g. ``mas_topologies.tasks.ml_architect.prompts.ML_PROMPT_SET``)
live with their task.
"""

from dataclasses import dataclass

from .protocol import FINAL_ANSWER_INSTRUCTION, VERDICT_INSTRUCTION

# ---------------------------------------------------------------------------
# Domain-neutral defaults.  The VERDICT and FINAL ANSWER instructions are not
# repeated here: they are appended automatically (see PromptSet.verdict_hint
# and PromptSet.final_answer_hint).
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_MESSAGE = (
    "You are an expert problem solver. Read the task carefully and produce a complete, "
    "correct and well-justified answer that satisfies every stated requirement.\n"
    "If you receive feedback, address every point explicitly and give your COMPLETE "
    "revised answer, not only the changes."
)

CRITIC_SYSTEM_MESSAGE = (
    "You are a careful, constructive reviewer. You will receive a task and a proposed "
    "answer to it.\n"
    "Check the answer for correctness, completeness, and whether it satisfies every "
    "requirement stated in the task.\n"
    "If the answer is fully acceptable, say so briefly. Otherwise list only the specific "
    "problems you found, one per line, and ask for the COMPLETE revised answer. "
    "Be concise and do not write the answer yourself."
)

ADVERSARIAL_CRITIC_SYSTEM_MESSAGE = (
    "You are an adversarial critic. You will receive a task and a proposed answer to it.\n"
    "Stress-test the answer: challenge its key assumptions and choices, look for failure "
    "modes, edge cases and counter-examples, and point out stronger alternatives.\n"
    "Raise real, substantive objections, not nitpicks, and ask for the COMPLETE revised "
    "answer. Keep challenging while substantive objections remain; approve only once they "
    "have been convincingly addressed."
)

JUDGE_SYSTEM_MESSAGE = (
    "You are an impartial expert judge. You will receive a task and several candidate "
    "answers produced independently.\n"
    "Compare them on correctness, completeness and how well they satisfy the task. Then "
    "produce ONE final answer: select the best candidate, or combine the strongest parts of "
    "several, fixing any errors you find.\n"
    "Be decisive: output a single complete answer, not a list of options."
)

REFLECTION_PROMPT = (
    "Critically review your previous answer against the task: is it correct, complete, and "
    "does it satisfy every requirement?\n"
    "List only the specific problems you find, one per line. If there are none, say so "
    "briefly. Do not rewrite the answer here."
)

JUDGE_TASK_TEMPLATE = (
    "## ORIGINAL TASK\n{context}\n\n"
    "## {n} CANDIDATE ANSWERS\n\n{proposals}"
    "## YOUR TASK\n"
    "Compare the {n} candidate answers above and produce ONE complete final answer to the "
    "original task."
)

REVISION_PROMPT = (
    "Based on your reflection above, provide your COMPLETE revised answer "
    "(the full answer, not only the changes)."
)

DEBATE_PROMPT = (
    "Here are the most recent answers of the other agents (debate round {round}):\n\n{others}"
    "Compare them with your own previous answer. Where the answers differ, identify the specific "
    "step or assumption that causes the difference and decide which version is correct. Change "
    "your answer only if you find a concrete error in your own reasoning; do not simply adopt the "
    "majority view. Be concise: give the decisive argument rather than repeating the whole "
    "derivation, then your COMPLETE updated answer."
)


@dataclass
class PromptSet:
    """Bundles all prompts needed to run a task through any topology.

    Attributes:
        system_message:      Primary agent role and output format. Required.
        critic_message:      Reviewer persona for ``peer_review`` topology.
        adversary_message:   Adversary persona for ``adversarial`` topology.
        judge_message:       Judge persona for ``parallel_ensemble`` topology.
        reflection_prompt:   Self-critique instructions for ``self_reflection``.
        judge_task_template: The Judge's task for ``parallel_ensemble``, with the
            placeholders ``{context}`` (original task), ``{n}`` (number of
            proposals) and ``{proposals}`` (all proposals, already formatted).
        revision_prompt:     Asks for the revised answer after each reflection
            in ``self_reflection``.
        debate_prompt:       Sent to each debater from round 2 of ``debate``, with
            the placeholders ``{others}`` (the other debaters' latest answers,
            already formatted) and ``{round}`` (the round number).

    Every role field defaults to a domain-neutral prompt, so a task only has
    to supply what it wants to specialise.
    """

    system_message: str
    critic_message: str = CRITIC_SYSTEM_MESSAGE
    adversary_message: str = ADVERSARIAL_CRITIC_SYSTEM_MESSAGE
    judge_message: str = JUDGE_SYSTEM_MESSAGE
    reflection_prompt: str = REFLECTION_PROMPT
    judge_task_template: str = JUDGE_TASK_TEMPLATE
    revision_prompt: str = REVISION_PROMPT
    debate_prompt: str = DEBATE_PROMPT
    code_execution_hint: str = ""
    """Appended to *system_message* when ``execute_code=True``.

    Leave empty (default) for pure text-generation tasks.  Set to a short
    instruction block when you want agents to write and run code, e.g.::

        PromptSet(
            system_message="...",
            code_execution_hint=(
                "When you have code to test, write it in a ```python block.\\n"
                "CodeRunner will execute it and post the output back.\\n"
                "Print concise summaries — not full arrays.\\n"
                "Say DONE when your final answer is complete."
            ),
        )
    """
    verdict_hint: str = VERDICT_INSTRUCTION
    """Appended to the critic, adversary and reflection prompts (unless they
    already contain ``VERDICT:``) so every critic ends with ``VERDICT: APPROVE``
    or ``VERDICT: REVISE``.  Only an APPROVE from the critic itself stops a run.
    Set to ``""`` to opt out (the run then always uses all *max_rounds*).
    """
    final_answer_hint: str = FINAL_ANSWER_INSTRUCTION
    """Appended to the generator's system message and the judge's message
    (unless they already contain the tag ``FINAL ANSWER:``) so the final message marks
    its answer with ``FINAL ANSWER:``.  Set to ``""`` to opt out.
    """


__all__ = [
    "PromptSet",
    "DEFAULT_SYSTEM_MESSAGE",
    "CRITIC_SYSTEM_MESSAGE",
    "ADVERSARIAL_CRITIC_SYSTEM_MESSAGE",
    "JUDGE_SYSTEM_MESSAGE",
    "REFLECTION_PROMPT",
    "JUDGE_TASK_TEMPLATE",
    "REVISION_PROMPT",
    "DEBATE_PROMPT",
]
