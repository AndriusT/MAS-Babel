<div align="center">

<img src="assets/MAS-babel-logo.png"
     alt="MAS-Babel logo"
     width="220">

</div>

<div align="center">

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2C%203.11%2C%203.12-blue.svg" alt="Python versions"></a>
<a href="https://github.com/AndriusT/MAS-Babel/stargazers"><img src="https://img.shields.io/github/stars/AndriusT/MAS-Babel?style=flat" alt="GitHub stars"></a>
<a href="https://github.com/AndriusT/MAS-Babel/network/members"><img src="https://img.shields.io/github/forks/AndriusT/MAS-Babel?style=flat" alt="GitHub forks"></a>
<a href="https://github.com/AndriusT/MAS-Babel/graphs/contributors"><img src="https://img.shields.io/github/contributors/AndriusT/MAS-Babel" alt="Contributors"></a>

</div>


<h1 align="center">MAS-Babel</h1>

<p align="center">

Multi-agent system (MAS) Babel is a tool for designing multi-agent architectures and orchestration patterns (sometimes referred to as *topologies*) for scientific applications. It helps you compose agent roles (e.g., *designer*, *critic*, *adversary*, *judge*, *self-critic* etc.), run topology-driven reasoning loops, and compare coordination patterns in a reproducible way. The main goal of MAS-Babel is an AutoGen-powered library of widely used multi-agent architectures that can be tested, bechmarked and further assembled into custom MAS pipelines for science applications.
</p>



## Implemented MAS Architectures

Currently implemented:

- **single**: a single agent baseline for testing.
- **self_reflection**: a single agent that critiques and revises its own output.
- **peer_review**: a two-agent system, where the *architect* generates an output and the *reviewer* provides feedback. 
- **adversarial**: a two-agent system, where the *architect* generates an output, which is then challenged by an *ardversary* agent for N rounds until a satisfactory output is generated.
- **parallel_ensemble**: multiple expert agents work in parallel, with the *judge* agent revieving and synthesising the final output.
- **debate**: several agents answer independently, then read each other's latest answers and revise over a few rounds, stopping early once they agree. The final answer is chosen by majority vote or by a *judge*. With a single round this is *self-consistency* (a majority vote over independent answers), a standard baseline at matched compute.

In progress: 

- **Planner-executor(s)**: a multi-agent system witha *planner* agent decompsing a task into subtasks with a single or multiple *executors* then executing the assigned tasks. 
- *...*

The implemented MAS architectures are illustrated below:

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Diagram of the six implemented MAS topologies: single agent, self-reflection, peer review, adversarial, parallel ensemble and debate" width="750" />
</p>



## Design Philosophy

Instead of hardcoding complex black-box MAS systems that is difficult to debug and modify, MAS-Babel provides the minimal building blocks that the end users can integrate into their custom pipelines. 

## Scientific Applications

MAS-Babel is designed to support scientific and research workflows where reliability, traceability, and iterative feedback matter. Typical use cases include:
- Scientific hypothesis formulation, data analysis, paper draft writing
- Literature review with a feedback loop to avoid halucinated content, references, etc. 
- Reproducing peer-reviewed research papers
- Machine learning algorithm optimization (e.g., iteratively choosing an optimal CNN architecture)


## Repository Contents

- src/mas_topologies/topologies/
  Core source code: multi-agent architecture (*topology*) implementation.

- src/mas_topologies/tasks/ml_architect/
  Task-specific wrapper for ML architecture design (context building and response parsing).

- examples/
  A number of example notebooks illustrating the core functionality of the library.

- tests/
  Unit tests for agent dispatch, context, logging, and topology behavior.


## Installation

Requirements:
- Python 3.10+

Install in editable mode:

    pip install -e .

Optional development dependencies:

    pip install -e ".[dev]"

## Quick Start

Run one of the demos at:

- examples/

You also need to register for API access with at least one LLM provider (e.g. [Anthropic](https://platform.claude.com/docs/en/api/overview#getting-api-keys), [Google Gemini](https://ai.google.dev/gemini-api/docs/api-key) or [OpenAI](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)) and either load your API keys into the environment or specify them directly in the code.

## Usage

A run needs a model client (any AutoGen chat-completion client), a `PromptSet` and a task string:

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from mas_topologies import PromptSet, TopologyAgent

generator = OpenAIChatCompletionClient(model="gpt-4o")
critic = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = TopologyAgent(
    model_client=generator,           # the generator (architect / experts)
    critic_model_client=critic,       # optional: a different model for the reviewer
    topology="peer_review",
    prompts=PromptSet(system_message="You are an expert physicist."),
    max_rounds=3,
    log_dir="logs",
)
result = asyncio.run(agent.run("Derive the period of a simple pendulum."))
print(result.final_answer, result.final_answer_tagged)
```

**Prompts.** Only `system_message` is required. Every other role (critic, adversary, judge, self-reflection, the judge's task text, the revision request and the debate prompt) has a domain-neutral default that you can override. `mas_topologies.tasks.ml_architect.ML_PROMPT_SET` is a complete example for ML architecture design.

**Models per role.** `model_client` drives the generator. `critic_model_client` (peer review), `adversary_model_client` (adversarial) and `judge_model_client` (parallel ensemble) each fall back to `model_client`, so you can pair different models, e.g. one model proposing and another critiquing. `expert_model_clients` (a list) gives the parallel experts or debaters different models: agent *i* uses entry *i*, cycling through the list if there are more agents than models.

**Round limits.** `max_rounds` is the maximum number of generator turns (proposals and revisions) in `peer_review`, `adversarial`, `self_reflection` and `debate` (turns per debater). If you leave it unset, the defaults are 4, 4, 3 and 3.

**Debate and voting.** `topology="debate"` uses `n_experts` debaters (default 3).
- From round 2, each debater sees the other debaters' latest answers (`PromptSet.debate_prompt`) and revises its own.
- The run stops early once all debaters' final answers agree. Set `stop_on_consensus=False` to always use every round.
- `aggregation="vote"` (the default) picks the majority answer, with ties going to the earliest debater. `aggregation="judge"` lets a judge synthesise the final answer instead, for code or free-form answers where exact voting makes no sense.
- Answers are compared with `normalize_answer`, which treats e.g. `80`, `80.0`, `$80$`, `\boxed{80}` and `80 km/h` as equal. Pass your own `answer_normalizer`, e.g. for symbolic equivalence.
- `max_rounds=1` gives self-consistency: independent answers plus a majority vote.
- `result.info` records the votes, any tie, the consensus round and each round's answers.

**When a run stops early.** Critic roles end each message with `VERDICT: APPROVE` or `VERDICT: REVISE`. This instruction is appended to their prompts automatically (`PromptSet.verdict_hint`). Only an `APPROVE` from the critic itself stops a run. The generator mentioning "approved" does not, and neither does text in the task.

**The final answer.** The generator (or the judge, in `parallel_ensemble`) is asked to mark its answer with `FINAL ANSWER:` (`PromptSet.final_answer_hint`). The answer is taken from the run's final message: the last generator message, or the judge's synthesis.
- `result.final_answer` is the text after the tag.
- If the tag is missing, `final_answer` is the whole final message and `result.final_answer_tagged` is `False`, so format failures can be counted rather than lost.
- `result.response_text` holds the full final message, and `result.messages` the whole conversation.
- To use a different answer format, pass your own `answer_extractor`.

**Code execution.** With `execute_code=True`, a `CodeRunner` agent executes the generator's Python blocks; code that a critic quotes is never executed. Single-agent and ensemble runs end when the generator says `DONE`, after any code in that message has run. The `DONE` message must contain the final answer. In `debate`, each debater's code is run after every round. In the next round, each debater sees its own execution output and the other debaters' answers together with their outputs. With `aggregation="judge"`, the judge also sees each final proposal's output.

**Token usage.** `result.usage` holds the run's token counts: `prompt_tokens`, `completion_tokens` and the number of model `calls`, in total and per role (`by_source`). Every model-generated message also carries its own token counts. Use these to compare topologies at matched compute.

**Logs.** With `log_dir` set, each run is appended as one JSON line, containing:
- every message, with its round number and token counts;
- the critics' `approved` flags;
- `final_answer` and `final_answer_tagged`;
- `usage` (as above) and `info` (topology-specific details, e.g. a debate's votes);
- any `metadata` you pass to `run()`. For model comparisons, include the model used for each role here.

## Roadmap

- Expand the MAS architecture catalog to cover additional orchestration patterns (e.g., a planner agent + multiple executor agents).
- Apply to scientific programming problem benchmarks, such as SciCode.
- Apply to science problem benchmarks, e.g., TPBench (theoretical physics problems)
- Apply to peer-reviewed paper reproduction benchmarks, e.g., Replicationbench.

## License

MIT