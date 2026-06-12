<p align="center">
  <img src="assets/MAS-babel-logo.png" alt="MAS Topologies Designer logo" width="220">
</p>

# MAS Topologies Designer

A multi-agent architecture design toolkit for machine learning. This repo helps you define dataset goals, build structured prompts for expert roles, and run topology-driven design pipelines that generate, review, and refine CNN model proposals.

## What this repo contains

- `src/`: core topology engine and task definitions
- `examples/ml_architect_demo.ipynb`: interactive demo for CIFAR-10 architecture design
- `examples/data/`: preserved placeholder directory for dataset caching without tracking large files

## Why use it

Use this project to prototype ML model design with collaborative AI roles like architect, reviewer, adversary, judge, and self-reflection. It is especially useful when you want a reproducible workflow for generating code, hyperparameters, and architecture rationale.

## Quick start

1. Install the package: `pip install -e .`
2. Open `examples/ml_architect_demo.ipynb`
3. Set your API key and run the demo cells

## Notes

- The repo preserves `examples/data/` in Git via a placeholder file while ignoring large dataset files
- The notebook uses `TopologyAgent` and `PromptSet` to orchestrate multi-agent design flows
