"""Tests for JSONL session logging."""

import json
import os

import pytest

from mas_topologies.logging import log_session
from mas_topologies.models import TopologyResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

METADATA = {
    "dataset": "MNIST",
    "input_shape": [28, 28, 1],
    "dataset_size": 60000,
    "iteration": 1,
}


def _result(messages=None, rounds_used=1) -> TopologyResult:
    if messages is None:
        messages = [{"source": "Architect", "content": "A nice architecture."}]
    return TopologyResult(
        response_text="A nice architecture.",
        messages=messages,
        rounds_used=rounds_used,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogSession:
    def test_noop_when_log_path_none(self, tmp_path):
        log_session(None, "single", _result(), METADATA)
        assert list(tmp_path.iterdir()) == []

    def test_creates_file(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result(), METADATA)
        assert os.path.exists(log_path)

    def test_appends_single_valid_json_line(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result(), METADATA)
        lines = open(log_path).readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])  # must not raise
        assert entry["topology"] == "single"
        assert entry["dataset"] == "MNIST"
        assert entry["input_shape"] == [28, 28, 1]
        assert entry["dataset_size"] == 60000
        assert entry["rounds_used"] == 1

    def test_appends_multiple_sessions_as_separate_lines(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result(), METADATA)
        log_session(log_path, "peer_review", _result(rounds_used=3), METADATA)
        lines = open(log_path).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["topology"] == "single"
        assert json.loads(lines[1])["topology"] == "peer_review"
        assert json.loads(lines[1])["rounds_used"] == 3

    def test_contains_iso_timestamp(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result(), METADATA)
        entry = json.loads(open(log_path).read())
        assert "timestamp" in entry
        assert "T" in entry["timestamp"]  # ISO 8601 format

    def test_metadata_merged_into_entry(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        meta = {"iteration": 5, "dataset": "CIFAR-10", "custom_field": "abc"}
        log_session(log_path, "single", _result(), meta)
        entry = json.loads(open(log_path).read())
        assert entry["iteration"] == 5
        assert entry["dataset"] == "CIFAR-10"
        assert entry["custom_field"] == "abc"

    def test_noop_metadata_is_optional(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result())  # no metadata
        entry = json.loads(open(log_path).read())
        assert entry["topology"] == "single"

    # --- approved flag ---

    def test_approved_true_when_reviewer_says_approved(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        msgs = [
            {"source": "Architect", "content": "My design"},
            {"source": "Reviewer", "content": "APPROVED"},
        ]
        log_session(log_path, "peer_review", _result(msgs))
        assert json.loads(open(log_path).read())["approved"] is True

    def test_approved_false_when_reviewer_does_not_approve(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        msgs = [
            {"source": "Architect", "content": "My design"},
            {"source": "Reviewer", "content": "Needs improvement."},
        ]
        log_session(log_path, "peer_review", _result(msgs))
        assert json.loads(open(log_path).read())["approved"] is False

    def test_approved_true_when_adversary_says_accepted(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        msgs = [
            {"source": "Architect", "content": "My design"},
            {"source": "Adversary", "content": "ACCEPTED"},
        ]
        log_session(log_path, "adversarial", _result(msgs))
        assert json.loads(open(log_path).read())["approved"] is True

    def test_approved_false_for_single_topology(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        log_session(log_path, "single", _result())
        assert json.loads(open(log_path).read())["approved"] is False

    # --- message structure ---

    def test_architect_messages_have_incrementing_rounds(self, tmp_path):
        log_path = str(tmp_path / "log.jsonl")
        msgs = [
            {"source": "Architect", "content": "round 1"},
            {"source": "Reviewer", "content": "needs work"},
            {"source": "Architect", "content": "round 2"},
            {"source": "Reviewer", "content": "APPROVED"},
        ]
        log_session(log_path, "peer_review", _result(msgs, rounds_used=2))
        entry = json.loads(open(log_path).read())
        arch_msgs = [m for m in entry["messages"] if m["source"] == "Architect"]
        assert arch_msgs[0]["round"] == 1
        assert arch_msgs[1]["round"] == 2

    def test_other_sources_included_in_messages(self, tmp_path):
        """Messages from any source (CodeRunner, Reflection, ...) are logged."""
        log_path = str(tmp_path / "log.jsonl")
        msgs = [
            {"source": "Architect", "content": "design"},
            {"source": "CodeRunner", "content": "output"},
        ]
        log_session(log_path, "single", _result(msgs))
        entry = json.loads(open(log_path).read())
        sources = [m["source"] for m in entry["messages"]]
        assert sources == ["Architect", "CodeRunner"]
        assert entry["messages"][1]["round"] == 1

