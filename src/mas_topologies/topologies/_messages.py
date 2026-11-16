"""Shared helpers for turning AutoGen messages into loggable dicts (with token usage)."""

from typing import Any, Dict, List, Optional, Sequence

from autogen_agentchat.messages import BaseChatMessage


def message_dict(message: BaseChatMessage, source: Optional[str] = None) -> Dict[str, Any]:
    """``{"source", "content"}`` for one chat message, plus its token usage.

    ``prompt_tokens`` / ``completion_tokens`` are added when the message came
    from a model call (AutoGen's ``models_usage``); CodeRunner output has none.
    """
    d: Dict[str, Any] = {"source": source or message.source, "content": message.to_text()}
    usage = getattr(message, "models_usage", None)
    if usage is not None:
        d["prompt_tokens"] = usage.prompt_tokens
        d["completion_tokens"] = usage.completion_tokens
    return d


def collect_messages(messages: Sequence) -> List[Dict[str, Any]]:
    """Return every agent chat message as ``{"source", "content", ...}`` dicts.

    The initial task message (``source="user"``) is skipped because it is the
    input, not part of the conversation.  Internal events (``BaseAgentEvent``)
    are skipped because their content is repeated in the chat messages.
    """
    return [
        message_dict(m)
        for m in messages
        if isinstance(m, BaseChatMessage) and m.source != "user"
    ]


def summarize_usage(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Token totals of a run: overall and per message source (role).

    Returns ``{"total": {...}, "by_source": {source: {...}}}`` where each entry
    has ``prompt_tokens``, ``completion_tokens`` and ``calls`` (model calls).
    """
    def empty() -> Dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    total, by_source = empty(), {}
    for m in messages:
        if "prompt_tokens" not in m:
            continue
        for bucket in (total, by_source.setdefault(m["source"], empty())):
            bucket["prompt_tokens"] += m["prompt_tokens"]
            bucket["completion_tokens"] += m["completion_tokens"]
            bucket["calls"] += 1
    return {"total": total, "by_source": by_source}


__all__ = ["message_dict", "collect_messages", "summarize_usage"]
