"""Output protocol shared by all topologies.

* **Verdicts** — critic roles (Reviewer, Adversary, self-reflection) end their
  message with a single line ``VERDICT: APPROVE`` or ``VERDICT: REVISE``.
  :func:`parse_verdict` is the only place that interprets it, so the stopping
  rule and the log's ``approved`` flag can never disagree.
* **Final answers** — the generator (or the Judge) marks its answer with
  ``FINAL ANSWER:``; :func:`extract_final_answer` returns what follows it.
"""

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from autogen_agentchat.conditions import FunctionalTermination
from autogen_agentchat.messages import BaseChatMessage

VERDICT_INSTRUCTION = (
    "End your message with exactly one final line: 'VERDICT: APPROVE' if no further "
    "changes are needed, or 'VERDICT: REVISE' if the proposal must be revised."
)

# Tolerates markdown around the verdict, e.g. "**VERDICT:** APPROVE"
_VERDICT = re.compile(
    r"VERDICT[*_`\s]*:[*_`\s]*(APPROVED?|ACCEPT(?:ED)?|REVISE)\b", re.IGNORECASE
)
_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.DOTALL)
# Bare keywords used by the original prompts; accepted only as the whole message
# (ignoring markdown, whitespace and trailing punctuation, e.g. "**APPROVED**")
_LEGACY_APPROVALS = {"APPROVED", "ACCEPTED", "REFLECTION_COMPLETE"}
_LEGACY_NOISE = re.compile(r"[\s*`#.!]+")


def parse_verdict(text: str) -> Optional[str]:
    """Return ``"APPROVE"``, ``"REVISE"`` or ``None`` (no verdict given).

    The last ``VERDICT:`` line outside code blocks wins.  A message that is
    exactly ``APPROVED`` / ``ACCEPTED`` / ``REFLECTION_COMPLETE`` also counts as
    approval (backward compatibility).  Mentions such as "NOT APPROVED" do not.
    """
    if _LEGACY_NOISE.sub("", text).upper() in _LEGACY_APPROVALS:
        return "APPROVE"
    matches = _VERDICT.findall(_CODE_BLOCK.sub("", text))
    if not matches:
        return None
    return "REVISE" if matches[-1].upper() == "REVISE" else "APPROVE"


def is_approval(text: str) -> bool:
    return parse_verdict(text) == "APPROVE"


def approval_termination(source: str) -> FunctionalTermination:
    """Stop a group chat when *source* (and only *source*) approves."""

    def _approved(messages) -> bool:
        return any(
            isinstance(m, BaseChatMessage) and m.source == source and is_approval(m.to_text())
            for m in messages
        )

    return FunctionalTermination(_approved)


def with_verdict_instruction(prompt: str, instruction: str = VERDICT_INSTRUCTION) -> str:
    """Append *instruction* to *prompt* unless it already asks for a verdict."""
    if not instruction or "VERDICT:" in prompt:
        return prompt
    return f"{prompt}\n\n{instruction}" if prompt else instruction


# ---------------------------------------------------------------------------
# Final answers
# ---------------------------------------------------------------------------

FINAL_ANSWER_INSTRUCTION = (
    "When your answer is complete, end your response with a line starting with "
    "'FINAL ANSWER:' followed by your complete final answer."
)

# Code-execution mode ends on the message that says DONE, and that message is the
# one the final answer is extracted from — so it must carry the whole answer.
DONE_FINAL_ANSWER_INSTRUCTION = (
    "The message in which you say DONE must itself contain your complete final answer, "
    "marked with 'FINAL ANSWER:' as instructed. Repeat the answer in full; do not refer "
    "back to earlier messages."
)

# Tolerates markdown around the tag, e.g. "**FINAL ANSWER:**" or "Final answer:"
_FINAL_TAG = re.compile(r"(?:\*\*|__)?FINAL[ _]ANSWER(?:\*\*|__)?\s*:(?:\*\*|__)?", re.IGNORECASE)
# A trailing DONE line (code-execution mode) is a control word, not part of the answer
_TRAILING_DONE = re.compile(r"\s*\bDONE\s*$")


def extract_final_answer(text: str) -> Tuple[str, bool]:
    """Return ``(answer, tagged)`` from the final message of a run.

    *answer* is the text after the last ``FINAL ANSWER:`` tag that is not
    inside a code block, and *tagged* is ``True``.  If there is no such tag (or
    nothing follows it), the whole message is returned with ``tagged=False`` so
    the run is kept but can be counted as a format failure.
    """
    code_spans = [m.span() for m in _CODE_BLOCK.finditer(text)]
    tags = [
        m for m in _FINAL_TAG.finditer(text)
        if not any(start <= m.start() < end for start, end in code_spans)
    ]
    if tags:
        tag = tags[-1].group(0)
        answer = _TRAILING_DONE.sub("", text[tags[-1].end():]).strip()
        # "**FINAL ANSWER: 80**": the tag opened a bold/underline marker that the
        # answer closes, so drop that closing marker too (-> "80", not "80**")
        for marker in ("**", "__"):
            if tag.count(marker) % 2 == 1 and answer.endswith(marker):
                answer = answer[: -len(marker)].rstrip()
        if answer:
            return answer, True
    return text.strip(), False


def with_final_answer_instruction(prompt: str, instruction: str = FINAL_ANSWER_INSTRUCTION) -> str:
    """Append *instruction* to *prompt* unless it already asks for the ``FINAL ANSWER:`` tag.

    Only the tag itself (upper case, with the colon) counts: a prompt that merely
    says "produce one final answer" still gets the instruction.
    """
    if not instruction or "FINAL ANSWER:" in prompt:
        return prompt
    return f"{prompt}\n\n{instruction}" if prompt else instruction


# ---------------------------------------------------------------------------
# Answer normalisation and voting (debate / self-consistency)
# ---------------------------------------------------------------------------

_BOXED = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_TEXT_CMD = re.compile(r"\\(?:text|mathrm|textrm)\{([^{}]*)\}")
_FRAC = re.compile(r"^\\[dt]?frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}$")
_ASSIGNMENT = re.compile(r"^[A-Za-z_\\][\w\\{}]*\s*=\s*")
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUMBER_WITH_UNIT = re.compile(rf"^({_NUMBER})\s*([A-Za-z°%][A-Za-z°%/\s^0-9\-·]*)?$")


def _to_number(s: str) -> Optional[float]:
    """Parse ``80``, ``1,234``, ``-1.5e3``, ``3/4`` or ``\\frac{3}{4}``; else ``None``."""
    s = s.strip()
    m = _FRAC.match(s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    if s.count("/") == 1:
        num, den = (_to_number(p) for p in s.split("/"))
        if num is None or den in (None, 0):
            return None
        return num / den
    if not re.fullmatch(_NUMBER, s) or not re.search(r"\d", s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def normalize_answer(text: str) -> str:
    """Canonical form of an extracted final answer, for comparing / voting.

    Conservative by design: removes formatting (markdown emphasis, ``$...$``,
    ``\\(...\\)``, ``\\boxed{...}``, ``\\text{...}``, a leading ``x =``, a
    trailing period), case-folds, and makes numbers comparable at 6 significant
    digits, so ``80``, ``80.0``, ``$80$``, ``\\boxed{80}`` and ``8e1`` agree.  A
    number followed only by a plain unit (``80 km/h``) is reduced to the number.
    No symbolic equivalence: pass your own normaliser for that (e.g. sympy-based).
    """
    s = (text or "").strip()
    for _ in range(2):   # markers can be nested, e.g. "**$80$**"
        s = s.strip().strip("*_`").strip()
        for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
                s = s[len(left):-len(right)].strip()
    m = _BOXED.search(s)
    if m:
        s = m.group(1)
    s = _TEXT_CMD.sub(r" \1", s).strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)

    candidate = _ASSIGNMENT.sub("", s)   # "x = 80" -> "80" (only kept if numeric)
    for c in (s, candidate):
        value = _to_number(c)
        if value is None:
            m = _NUMBER_WITH_UNIT.match(c)
            if m and m.group(2) and re.search(r"\d", m.group(1)):
                value = _to_number(m.group(1))
        if value is not None:
            return "0" if value == 0 else f"{value:.6g}"
    return s.lower()


def majority_vote(
    answers: List[str], normalizer: Callable[[str], str] = normalize_answer
) -> Tuple[int, Dict[str, int], bool]:
    """Majority vote over extracted answers.

    Returns ``(winner_index, counts, tie)`` where *counts* maps each normalised
    answer to its number of votes.  On a tie, the earliest agent (lowest index)
    among the tied answers wins and *tie* is ``True``.
    """
    keys = [normalizer(a) for a in answers]
    counts = dict(Counter(keys))
    best = max(counts.values())
    leaders = {k for k, n in counts.items() if n == best}
    winner = next(i for i, k in enumerate(keys) if k in leaders)
    return winner, counts, len(leaders) > 1


__all__ = [
    "VERDICT_INSTRUCTION",
    "parse_verdict",
    "is_approval",
    "approval_termination",
    "with_verdict_instruction",
    "FINAL_ANSWER_INSTRUCTION",
    "DONE_FINAL_ANSWER_INSTRUCTION",
    "extract_final_answer",
    "with_final_answer_instruction",
    "normalize_answer",
    "majority_vote",
]
