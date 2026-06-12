"""
ml_architect_demo.py
====================
Demonstrates mas_topologies applied to ML architecture design for CIFAR-10.

This script shows the full pipeline explicitly:
  1.  Define a PromptSet (architect + reviewer/adversary/judge/reflection prompts)
  2.  Build a TopologyAgent with your chosen topology
  3.  Prepare a context string from a DesignInput
  4.  Run the agent and parse the structured response

Provider selection (first matching env var wins):
    export OPENAI_API_KEY="sk-..."            # OpenAI
    export ANTHROPIC_API_KEY="sk-ant-..."     # Anthropic Claude

Usage:
    PYTHONPATH=src python examples/ml_architect_demo.py [--provider openai|anthropic] [--topology single|peer_review|adversarial|parallel_ensemble|self_reflection|all]

Examples:
    PYTHONPATH=src python examples/ml_architect_demo.py
    PYTHONPATH=src python examples/ml_architect_demo.py --provider anthropic --topology adversarial
    PYTHONPATH=src python examples/ml_architect_demo.py --topology peer_review
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, "/home/axt807/AutoGen/mas_topologies_designer/src")

from mas_topologies import TopologyAgent, PromptSet
from mas_topologies.tasks.ml_architect.models import DesignInput
from mas_topologies.tasks.ml_architect.context import prepare_context, parse_response

# Build fence so literal ``` never appears in this source file (used in prompts)
_F = chr(96) * 3

# ---------------------------------------------------------------------------
# Prompts — written out explicitly so the full prompt is visible in one place
# ---------------------------------------------------------------------------

ARCHITECT_PROMPT = f"""You are an expert ML architect specialising in deep learning CNN design.
Your role is to analyse datasets and propose optimal neural network architectures.

==============================================================================
CRITICAL: DATA FORMAT AND INPUT SHAPE HANDLING
==============================================================================

When you receive dataset information, you will see:
1. **input_shape**: The shape passed to create_model() — always HWC format
   (height, width, channels).  PyTorch models receive HWC and convert to BCHW
   internally.  Example: (28, 28, 1) = 28×28 grayscale.
2. **Value Range**: Important for preprocessing.

==============================================================================
YOUR ANALYSIS PROCESS
==============================================================================

1. Extract key properties — spatial dims, channels, value range, dataset size,
   framework (PyTorch vs TensorFlow).
2. Incorporate feedback — when prior training results are provided, use them to
   make targeted improvements.
3. Propose architecture — match literature for similar tasks; start the first
   Conv layer with C (channels from input_shape).

3a. Pooling decision — decide upfront how many pooling ops the architecture
    uses. State explicitly: "This architecture uses N pooling operations."
    Standard MaxPool2d(2) halves each spatial dimension.

3b. Flatten size — BEFORE writing nn.Linear(), calculate:
      flatten_size = final_channels × (H // reduction_factor)
                                    × (W // reduction_factor)
    where reduction_factor = 2^(number of pooling operations).
    Use variables, never magic numbers.

4. Justify choices — explain parameter count vs dataset size, regularisation
   strategy, and verify pooling consistency.
5. Specify hyperparameters — optimizer, learning_rate, batch_size, epochs,
   learning_rate_schedule, warmup_epochs, dropout_rate, weight_decay.
6. Preprocessing strategy — normalisation, augmentation.
7. Training strategy — train/val/test split, metrics, early stopping.
8. Implementation code — complete, executable Python in a code block.

==============================================================================
IMPLEMENTATION REQUIREMENTS (PYTORCH)
==============================================================================

def create_model(input_shape, num_classes) -> nn.Module:
    H, W, C = input_shape   # always unpack like this

Rules:
- First Conv2d MUST use C:  nn.Conv2d(C, 32, kernel_size=3, padding=1)
- Flatten size MUST be parametric:  128 * (H // 4) * (W // 4)  NOT  128*8*8
- forward() MUST return raw logits (no softmax/sigmoid)
- return an instantiated model, not a class

==============================================================================
REQUIRED OUTPUT FORMAT
==============================================================================

Your response MUST contain all three sections in order:

1. **Analysis Section** — dataset properties, design rationale, strategy.
2. **Implementation Section** — complete Python in a code block:

{_F}python
import torch, torch.nn as nn, torch.nn.functional as F

def create_model(input_shape, num_classes):
    H, W, C = input_shape
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
            # ... more layers ...
            self.fc_out = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            # add shape comments at key steps
            return self.fc_out(x)   # raw logits
    return Model()
{_F}

3. **Hyperparameters Section** — JSON in a code block (no comments, no trailing commas):

{_F}json
{{
  "hyperparameters": {{
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate_schedule": "none",
    "warmup_epochs": 0,
    "dropout_rate": 0.25
  }}
}}
{_F}
"""

CRITIC_PROMPT = (
    "You are an expert ML architect and code reviewer specialising in PyTorch CNN implementations.\n"
    "You will receive a proposed architecture alongside the original dataset task.\n\n"
    "=== SECTION A: Code Correctness ===\n"
    "1. Does flatten_size use H and W variables (not magic numbers)?\n"
    "2. Does the first Conv2d use C from 'H, W, C = input_shape' (not hardcoded)?\n"
    "3. Does create_model() return an instantiated model (not a class)?\n"
    "4. Does forward() return raw logits (no softmax/sigmoid)?\n"
    "5. Is a JSON hyperparameters block present?\n\n"
    "=== SECTION B: Design Quality ===\n"
    "6. Model capacity vs dataset size — is it appropriate? Flag obvious over/under-parameterisation.\n"
    "7. Spatial validity — after all pooling, are spatial dims still >1×1?\n"
    "8. Architecture–task fit — does the style match the task (e.g. conv layers for images)?\n"
    "9. Regularisation — is it adequate given the dataset size?\n"
    "10. Hyperparameter plausibility — are values in sane ranges?\n\n"
    "=== OUTPUT RULES ===\n"
    "If ALL checks pass: respond with exactly APPROVED.\n"
    "Otherwise: list only the specific issues found, one per line, prefixed with the check number.\n"
    "Do not rewrite the architecture. End with:\n"
    "Please provide your COMPLETE revised response, including the full Analysis, Python code block, "
    "and JSON hyperparameters block."
)

ADVERSARY_PROMPT = (
    "You are an adversarial ML architecture critic. Your role is to challenge and stress-test proposals.\n\n"
    "=== ADVERSARIAL CRITIQUE STRATEGY ===\n"
    "Question every major design choice. Play devil's advocate.\n"
    "Challenge trade-offs: Is the complexity justified? Could simpler work?\n"
    "Point out potential failure modes. Suggest significantly different alternatives.\n"
    "Probe hyperparameters: Are these really optimal?\n\n"
    "=== FOCUS AREAS ===\n"
    "1. Architecture depth/width — is it overkill or underpowered?\n"
    "2. Pooling strategy — too aggressive?\n"
    "3. Regularisation — sufficient? Missing alternatives (weight decay, data aug)?\n"
    "4. Hyperparameters — truly optimal for this dataset?\n"
    "5. Computational efficiency — could a simpler design work just as well?\n\n"
    "=== OUTPUT RULES ===\n"
    "Start with: CHALLENGE: [main objection]\n"
    "List 3–5 specific challenge points as bullet points.\n"
    "End with: Please provide your COMPLETE revised response, including full Analysis, "
    "Python code block, and JSON hyperparameters.\n"
    "Never respond with APPROVED."
)

JUDGE_PROMPT = (
    "You are a world-class ML architecture synthesis expert.\n"
    "You will receive multiple architecture proposals from different expert architects.\n"
    "Your role is to analyse, compare, and synthesise the BEST proposal.\n\n"
    "=== COMPARISON CRITERIA ===\n"
    "1. Code correctness — parametric flatten, raw logits, proper signatures\n"
    "2. Architectural trade-offs — depth, width, pooling strategy\n"
    "3. Capacity fit — model size vs dataset size\n"
    "4. Regularisation strategy — dropout, weight decay, batch size\n"
    "5. Hyperparameter balance — learning rate, epochs, optimiser\n"
    "6. Simplicity and robustness\n\n"
    "=== OUTPUT ===\n"
    "Provide a COMPLETE, FINAL proposal:\n"
    "1. Analysis Section — comparative summary and selection rationale\n"
    "2. Python code block — complete create_model(input_shape, num_classes)\n"
    "3. JSON hyperparameters block — final training specs\n"
    "Be decisive. Output one synthesised architecture, ready for training."
)

REFLECTION_PROMPT = (
    "Review your architecture proposal above against these checks:\n\n"
    "=== OUTPUT FORMAT ===\n"
    "1. Does the response contain all three sections: Analysis, Python code block, JSON block?\n"
    "2. Is there exactly ONE Python code block?\n"
    "3. Does the Python code block contain all necessary imports?\n\n"
    "=== CODE CORRECTNESS ===\n"
    "4. Correct signature: create_model(input_shape, num_classes)?\n"
    "5. Flatten size uses H and W variables (not magic numbers)?\n"
    "6. First Conv layer uses C from input_shape?\n"
    "7. Output layer sized to num_classes?\n"
    "8. Returns an instantiated model?\n"
    "9. forward() returns raw logits?\n"
    "10. JSON block valid (no comments, no trailing commas)?\n\n"
    "=== DESIGN QUALITY ===\n"
    "11. Model capacity appropriate for dataset size?\n"
    "12. Spatial dimensions valid after all pooling ops (>1×1)?\n"
    "13. Regularisation adequate given dataset size?\n"
    "14. Hyperparameter values in sane ranges?\n\n"
    "If ALL checks pass: respond with exactly REFLECTION_COMPLETE.\n"
    "Otherwise: list only the issues found, one per line, prefixed with check number."
)

# Assemble into a PromptSet — the generic contract TopologyAgent understands
PROMPT_SET = PromptSet(
    system_message=ARCHITECT_PROMPT,
    critic_message=CRITIC_PROMPT,
    adversary_message=ADVERSARY_PROMPT,
    judge_message=JUDGE_PROMPT,
    reflection_prompt=REFLECTION_PROMPT,
)

# ---------------------------------------------------------------------------
# Design task — CIFAR-10 CNN
# ---------------------------------------------------------------------------

CIFAR10_INPUT = DesignInput(
    dataset_name="CIFAR-10",
    dataset_description=(
        "60 000 colour images (32×32 px, 3 channels) spread across 10 object "
        "classes: airplane, automobile, bird, cat, deer, dog, frog, horse, "
        "ship, truck.  Standard 50k / 10k train/test split."
    ),
    num_classes=10,
    input_shape=(32, 32, 3),
    dataset_size=60_000,
    framework="PyTorch",
)

ALL_TOPOLOGIES = [
    "single",
    "peer_review",
    "adversarial",
    "parallel_ensemble",
    "self_reflection",
]

# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _build_client(provider: str):
    """Return (model_client, label) or (None, None) if no key is available."""
    if provider in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIChatCompletionClient(
            model=model, api_key=os.environ["OPENAI_API_KEY"]
        ), f"openai/{model}"

    if provider in ("anthropic", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        return AnthropicChatCompletionClient(
            model=model, api_key=os.environ["ANTHROPIC_API_KEY"]
        ), f"anthropic/{model}"

    return None, None

# ---------------------------------------------------------------------------
# Dry-run mock helpers
# ---------------------------------------------------------------------------

_MOCK_JSON = json.dumps({
    "hyperparameters": {
        "optimizer": "adam", "learning_rate": 0.001, "batch_size": 128,
        "epochs": 50, "weight_decay": 1e-4, "learning_rate_schedule": "cosine",
        "warmup_epochs": 5, "dropout_rate": 0.3,
    }
})
_MOCK_RESPONSE = (
    "## Architecture Proposal\n\n"
    "A compact ResNet-style network with three residual blocks (64, 128, 256 "
    "channels), global average pooling, and a single fully-connected head.\n\n"
    f"```json\n{_MOCK_JSON}\n```\n"
)

_TOPOLOGY_PATCH_TARGETS = {
    t: f"mas_topologies.agent.run_{t}" for t in ALL_TOPOLOGIES
}

# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

async def run_topology(topology: str, model_client, log_dir: str, dry_run: bool = False):
    print(f"\n{'='*60}")
    print(f"  Topology: {topology.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    agent = TopologyAgent(
        model_client=model_client,
        topology=topology,
        prompts=PROMPT_SET,
        log_dir=log_dir,
    )
    context = prepare_context(CIFAR10_INPUT)
    metadata = {
        "dataset": CIFAR10_INPUT.dataset_name,
        "input_shape": list(CIFAR10_INPUT.input_shape),
        "num_classes": CIFAR10_INPUT.num_classes,
        "dataset_size": CIFAR10_INPUT.dataset_size,
    }

    if dry_run:
        from unittest.mock import AsyncMock, patch
        from mas_topologies.models import TopologyResult
        mock_result = TopologyResult(
            response_text=_MOCK_RESPONSE,
            messages=[{"source": "Architect", "content": "mock"}],
            rounds_used=1,
        )
        with patch(_TOPOLOGY_PATCH_TARGETS[topology], new_callable=AsyncMock, return_value=mock_result):
            result = await agent.run(context, metadata=metadata)
    else:
        result = await agent.run(context, metadata=metadata)

    output = parse_response(result.response_text, CIFAR10_INPUT)

    print(f"\nOptimizer      : {output.hyperparameters.get('optimizer', 'N/A')}")
    print(f"Learning rate  : {output.hyperparameters.get('learning_rate', 'N/A')}")
    print(f"Batch size     : {output.hyperparameters.get('batch_size', 'N/A')}")
    print(f"Epochs         : {output.hyperparameters.get('epochs', 'N/A')}")
    print(f"LR schedule    : {output.hyperparameters.get('learning_rate_schedule', 'N/A')}")
    print(f"\nProposal excerpt:\n{output.proposed_architecture[:400].strip()}")
    return output

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="mas_topologies ML architecture design demo (CIFAR-10)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "auto"],
        default="auto",
        help="LLM provider to use (default: auto-detect from env vars)",
    )
    parser.add_argument(
        "--topology",
        choices=ALL_TOPOLOGIES + ["all"],
        default="all",
        help="Topology to run (default: all)",
    )
    args = parser.parse_args()

    model_client, provider_label = _build_client(args.provider)
    dry_run = model_client is None
    if dry_run:
        from unittest.mock import MagicMock
        model_client = MagicMock()
        print("[WARNING] No API key found — running in dry-run mode (responses are mocked).")
    else:
        print(f"[INFO] Provider: {provider_label}")

    topologies = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]

    log_dir = Path("examples/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nmas_topologies — ML Architecture Design Demo")
    print(f"Task      : CNN for {CIFAR10_INPUT.dataset_name} ({CIFAR10_INPUT.num_classes} classes, {CIFAR10_INPUT.input_shape})")
    print(f"Mode      : {'DRY RUN (mocked)' if dry_run else f'LIVE ({provider_label})'}")
    print(f"Topologies: {', '.join(topologies)}")

    results = {}
    for topology in topologies:
        output = await run_topology(topology, model_client, str(log_dir), dry_run=dry_run)
        results[topology] = output.hyperparameters

    print(f"\n{'='*60}")
    print("  Summary — hyperparameters by topology")
    print(f"{'='*60}")
    print(f"{'Topology':<22} {'Optimizer':<12} {'LR':<8} {'BS':<6} {'Epochs'}")
    print("-" * 60)
    for t, hp in results.items():
        print(
            f"{t:<22} {hp.get('optimizer','?'):<12} "
            f"{hp.get('learning_rate','?'):<8} "
            f"{hp.get('batch_size','?'):<6} "
            f"{hp.get('epochs','?')}"
        )

    log_files = list(log_dir.glob("session_*.jsonl"))
    if log_files:
        print(f"\nSession logs written to {log_dir}/ ({len(log_files)} file(s)).")


if __name__ == "__main__":
    asyncio.run(main())
