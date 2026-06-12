"""Tests for context building and response parsing (no API calls needed)."""

import json

import pytest

from mas_topologies.tasks.ml_architect.context import (
    _prepare_feedback_section,
    parse_response,
    prepare_context,
)
from mas_topologies.tasks.ml_architect.models import DataFormatInfo, DesignInput, DesignOutput

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASIC_INPUT = DesignInput(
    dataset_name="CIFAR-10",
    dataset_description="Natural images across 10 categories",
    num_classes=10,
    input_shape=(32, 32, 3),
    dataset_size=50000,
)

_FEEDBACK = {
    "final_val_accuracy": 0.85,
    "best_val_accuracy": 0.87,
    "final_train_accuracy": 0.95,
    "overfitting_gap": 0.10,
    "convergence_speed": "fast",
    "performance_rating": "good",
    "issues": ["overfitting detected"],
    "recommendations": ["add dropout"],
    "next_steps": ["reduce learning rate"],
}


# ---------------------------------------------------------------------------
# prepare_context
# ---------------------------------------------------------------------------

class TestPrepareContext:
    def test_contains_dataset_name(self):
        assert "CIFAR-10" in prepare_context(BASIC_INPUT)

    def test_contains_input_shape(self):
        assert "(32, 32, 3)" in prepare_context(BASIC_INPUT)

    def test_contains_num_classes(self):
        assert "10" in prepare_context(BASIC_INPUT)

    def test_contains_dataset_description(self):
        assert "Natural images" in prepare_context(BASIC_INPUT)

    def test_contains_dataset_size(self):
        assert "50000" in prepare_context(BASIC_INPUT)

    def test_no_feedback_section_on_iteration_1(self):
        assert "PREVIOUS ITERATION FEEDBACK" not in prepare_context(BASIC_INPUT)

    def test_feedback_section_on_iteration_2(self):
        inp = BASIC_INPUT.model_copy(
            update={"iteration_number": 2, "previous_feedback": _FEEDBACK}
        )
        ctx = prepare_context(inp)
        assert "PREVIOUS ITERATION FEEDBACK" in ctx
        assert "overfitting detected" in ctx
        assert "add dropout" in ctx

    def test_no_feedback_when_iteration_2_but_no_feedback_dict(self):
        inp = BASIC_INPUT.model_copy(update={"iteration_number": 2})
        assert "PREVIOUS ITERATION FEEDBACK" not in prepare_context(inp)

    def test_data_format_info_included(self):
        info = DataFormatInfo(
            shape=(64, 3, 32, 32),
            dtype="float32",
            data_format="BCHW",
            value_range=(0.0, 1.0),
            num_samples=50000,
            num_classes=10,
        )
        ctx = prepare_context(BASIC_INPUT.model_copy(update={"data_format_info": info}))
        assert "BCHW" in ctx
        assert "0.000" in ctx

    def test_framework_uppercased(self):
        inp = BASIC_INPUT.model_copy(
            update={"framework": "pytorch", "framework_version": "2.1.0"}
        )
        ctx = prepare_context(inp)
        assert "PYTORCH" in ctx
        assert "2.1.0" in ctx

    def test_constraints_included(self):
        inp = BASIC_INPUT.model_copy(update={"constraints": {"max_params": "5M"}})
        ctx = prepare_context(inp)
        assert "max_params" in ctx
        assert "5M" in ctx

    def test_user_notes_included(self):
        inp = BASIC_INPUT.model_copy(update={"user_notes": "Prefer BatchNorm"})
        assert "Prefer BatchNorm" in prepare_context(inp)

    def test_initial_architecture_included(self):
        inp = BASIC_INPUT.model_copy(update={"initial_architecture": "ResNet-style"})
        assert "ResNet-style" in prepare_context(inp)

    def test_papers_included_and_truncated_at_5000(self):
        long_paper = "A" * 10_000
        inp = BASIC_INPUT.model_copy(update={"papers_content": [long_paper]})
        ctx = prepare_context(inp)
        assert "A" * 5_000 in ctx
        assert "A" * 5_001 not in ctx

    def test_returns_string(self):
        assert isinstance(prepare_context(BASIC_INPUT), str)


# ---------------------------------------------------------------------------
# _prepare_feedback_section
# ---------------------------------------------------------------------------

class TestPrepareFeedbackSection:
    def test_formats_accuracy_as_percent(self):
        section = _prepare_feedback_section({"final_val_accuracy": 0.912})
        assert "91.20%" in section

    def test_best_accuracy_formatted(self):
        section = _prepare_feedback_section({"best_val_accuracy": 0.75})
        assert "75.00%" in section

    def test_no_issues_shows_none_message(self):
        section = _prepare_feedback_section({})
        assert "None" in section

    def test_issues_listed(self):
        section = _prepare_feedback_section(
            {"issues": ["gradient explosion", "slow convergence"]}
        )
        assert "gradient explosion" in section
        assert "slow convergence" in section

    def test_recommendations_listed(self):
        section = _prepare_feedback_section(
            {"recommendations": ["reduce lr", "add weight decay"]}
        )
        assert "reduce lr" in section
        assert "add weight decay" in section

    def test_next_steps_listed(self):
        section = _prepare_feedback_section({"next_steps": ["use cosine schedule"]})
        assert "use cosine schedule" in section

    def test_performance_rating_uppercased(self):
        section = _prepare_feedback_section({"performance_rating": "poor"})
        assert "POOR" in section

    def test_returns_string(self):
        assert isinstance(_prepare_feedback_section({}), str)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def _resp(self, hp: dict) -> str:
        """Build a minimal valid response with a JSON hyperparameters block."""
        return f"Analysis text.\n```json\n{json.dumps({'hyperparameters': hp})}\n```\n"

    def test_returns_design_output(self):
        out = parse_response(self._resp({}), BASIC_INPUT)
        assert isinstance(out, DesignOutput)

    def test_proposed_architecture_is_full_text(self):
        out = parse_response("some text", BASIC_INPUT)
        assert out.proposed_architecture == "some text"

    def test_extracts_learning_rate(self):
        out = parse_response(self._resp({"learning_rate": 0.005}), BASIC_INPUT)
        assert out.hyperparameters["learning_rate"] == 0.005

    def test_extracts_batch_size(self):
        out = parse_response(self._resp({"batch_size": 128}), BASIC_INPUT)
        assert out.hyperparameters["batch_size"] == 128

    def test_extracts_epochs(self):
        out = parse_response(self._resp({"epochs": 50}), BASIC_INPUT)
        assert out.hyperparameters["epochs"] == 50

    def test_extracts_optimizer(self):
        out = parse_response(self._resp({"optimizer": "sgd"}), BASIC_INPUT)
        assert out.hyperparameters["optimizer"] == "sgd"

    def test_always_includes_framework_from_input(self):
        inp = BASIC_INPUT.model_copy(update={"framework": "pytorch"})
        out = parse_response("no json", inp)
        assert out.hyperparameters["framework"] == "pytorch"

    def test_always_includes_input_shape(self):
        out = parse_response("no json", BASIC_INPUT)
        assert out.hyperparameters["input_shape"] == (32, 32, 3)

    def test_always_includes_num_classes(self):
        out = parse_response("no json", BASIC_INPUT)
        assert out.hyperparameters["num_classes"] == 10

    def test_missing_json_block_returns_defaults(self, capsys):
        out = parse_response("no json block here", BASIC_INPUT)
        assert isinstance(out, DesignOutput)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_uses_last_json_block_when_multiple(self):
        first = json.dumps({"hyperparameters": {"learning_rate": 0.1}})
        last = json.dumps({"hyperparameters": {"learning_rate": 0.0005}})
        resp = f"```json\n{first}\n```\nsome text\n```json\n{last}\n```\n"
        out = parse_response(resp, BASIC_INPUT)
        assert out.hyperparameters["learning_rate"] == 0.0005

    def test_strips_inline_comments_from_json(self):
        resp = '```json\n{"hyperparameters": {"learning_rate": 0.001 // the lr\n}}\n```\n'
        out = parse_response(resp, BASIC_INPUT)
        assert out.hyperparameters["learning_rate"] == 0.001

    def test_default_optimizer_when_key_absent(self):
        out = parse_response(self._resp({}), BASIC_INPUT)
        assert out.hyperparameters["optimizer"] == "adam"

    def test_default_dropout_rate_when_key_absent(self):
        out = parse_response(self._resp({}), BASIC_INPUT)
        assert out.hyperparameters["dropout_rate"] == 0.0

