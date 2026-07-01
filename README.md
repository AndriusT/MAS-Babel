<table>
<tr>
<td width="260" valign="top">
  <img src="assets/MAS-babel-logo.png" alt="MAS Topologies Designer logo" width="220">
</td>
<td valign="top">


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/badge/python-3.10%2C%203.11%2C%203.12-blue.svg)](https://www.python.org/)

# MAS Topologies Designer

MAS-Babel is a library of reusable multi-agent system (MAS) topologies for AI workflows. It helps you compose agent roles (designer, critic, adversary, judge, self-reflector), run topology-driven reasoning loops, and compare coordination patterns in a reproducible way.

</td>
</tr>
</table>

## What Is MAS-Babel?

MAS-Babel is a topology library for multi-agent AI orchestration.

In this context, a topology means how agents are organized and coordinated to solve a task. Examples include:
- single expert baseline
- designer plus critic iterative review
- adversarial challenge loops
- parallel experts with a coordinator or judge
- self-reflection and revision

The goal is to collect, standardize, and benchmark commonly used MAS patterns so teams can test and reuse proven architectures instead of rebuilding orchestration logic from scratch.

## Why Use It?

- Reusable MAS patterns: switch topologies without rewriting task logic.
- Clear role separation: prompts and responsibilities are explicit per agent role.
- Reproducible experimentation: consistent interfaces, structured outputs, and logging.
- Science-oriented by design: suitable for research and scientific pipelines, while remaining general enough for broader AI use cases.

## Core Topologies

Current built-in topologies include:

- single: one-agent baseline for fast iteration.
- peer_review: primary agent with critic feedback loops.
- adversarial: primary agent with adversarial challenge loops.
- parallel_ensemble: multiple experts in parallel, then judge synthesis.
- self_reflection: one agent that critiques and revises its own output.

## Repository Contents

- src/mas_topologies/
  Core library: topology dispatcher, prompt contracts, models, logging, and topology implementations.

- src/mas_topologies/topologies/
  Topology execution modules (single, peer review, adversarial, parallel ensemble, self reflection).

- src/mas_topologies/tasks/ml_architect/
  Task-specific wrapper for ML architecture design (context building and response parsing).

- examples/ml_architect_demo.ipynb
  Interactive notebook demo.

- examples/self_reflection.ipynb
  Example focused on self-reflection workflow.

- examples/ml_architect_demo.py
  Scripted demo runner for topology comparisons.

- examples/data/
  Placeholder directory for local dataset caching (large files not tracked).

- tests/
  Unit tests for agent dispatch, context, logging, and topology behavior.

- assets/
  Project images and branding assets.

## Installation

Requirements:
- Python 3.10+

Install in editable mode:

    pip install -e .

Optional development dependencies:

    pip install -e ".[dev]"

## Quick Start

1. Set a model provider key (for example OPENAI_API_KEY or ANTHROPIC_API_KEY).
2. Install the package.
3. Run one of the demos:

Notebook path:
- examples/ml_architect_demo.ipynb

Script path:
- examples/ml_architect_demo.py

Example command:

    PYTHONPATH=src python examples/ml_architect_demo.py --topology single

## Minimal Workflow

1. Define role prompts with PromptSet.
2. Create a TopologyAgent and choose a topology.
3. Build task context (plain text or task-specific schema).
4. Run the topology asynchronously.
5. Parse the final response into your task output format.

This separation keeps orchestration generic and task logic modular.

## Scientific Applications

MAS-Babel is designed to support scientific and research workflows where reliability, traceability, and iterative critique matter. Typical use cases include:
- model and experiment design assistance
- method comparison through structured debate
- evidence synthesis across multiple expert agents
- iterative proposal refinement with explicit review stages

Although science is a primary target, the framework is domain-agnostic and can be adapted to many AI planning and generation tasks.

## Roadmap

- Expand the topology catalog to cover additional orchestration patterns.
- Add comparative evaluation utilities across topologies.
- Add more task templates beyond ML architecture design.
- Improve docs and examples for domain-specific adaptation.

## License

MIT