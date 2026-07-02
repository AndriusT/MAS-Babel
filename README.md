<table>
<tr>
<td width="260" valign="top">
  <img src="assets/MAS-babel-logo.png" alt="MAS Topologies Designer logo" width="220">
</td>
<td valign="top">


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/badge/python-3.10%2C%203.11%2C%203.12-blue.svg)](https://www.python.org/)

# MAS-Babel

Multi-agent system (MAS) Babel is a tool for designing multi-agent architectures and orchestration patterns (sometimes referred to as *topologies*) for scientific applications. It helps you compose agent roles (e.g., *designer*, *critic*, *adversary*, *judge*, *self-critic* etc.), run topology-driven reasoning loops, and compare coordination patterns in a reproducible way. The main goal of MAS-Babel is and AutoGen-powered library of widely used multi-agent architectures that can be tested, bechmarked, assembled into custom MAS pipelines for science applications.

</td>
</tr>
</table>

## Implemented MAS Architectures

Currently implemented:

- **single**: a single agent baseline for testing.
- **self_reflection**: a single agent that critiques and revises its own output.
- **peer_review**: a two-agent system, where the *architect* generates an output and the *reviewer* provides feedback. 
- **adversarial**: a two-agent system, where the *architect* generates an output, which is then challenged by an *ardversary* agent for N rounds until a satisfactory output is generated.
- **parallel_ensemble**: multiple expert agents work in parallel, with the *judge* agent revieving and synthesising the final output.

In progress: 

- **Planner-executor(s)**: a multi-agent system witha *planner* agent decompsing a task into subtasks with a single or multiple *executors* then executing the assigned tasks. 
- **Agent ensemble**: multiple expert agents generate outputs with the final output chosen by majority vote.
- *...*


## Design Philosophy

Instead of hardcoding a complex black-box MAS system that is difficult to debug and modify, MAS-Babel provides the minimal building blocks that the end users can integrate into their custom pipelines. 

## Scientific Applications

MAS-Babel is designed to support scientific and research workflows where reliability, traceability, and iterative critique matter. Typical use cases include:
- model and experiment design assistance
- method comparison through structured debate
- evidence synthesis across multiple expert agents
- iterative proposal refinement with explicit review stages

Although science is a primary target, the framework is domain-agnostic and can be adapted to many AI planning and generation tasks.
 

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

Run one of the demos:

Notebook path:
- examples/ml_architect_demo.ipynb

Script path:
- examples/ml_architect_demo.py


## Roadmap

- Expand the topology catalog to cover additional orchestration patterns.
- Add comparative evaluation utilities across topologies.
- Add more task templates beyond ML architecture design.
- Improve docs and examples for domain-specific adaptation.

## License

MIT