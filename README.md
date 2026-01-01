# Misguided Attention

A benchmark for evaluating LLM reasoning in the presence of misleading information. The prompts are variations of well-known thought experiments, riddles, and paradoxes designed to test whether models rely on pattern matching from training data versus genuine logical deduction.

## Overview

Many LLMs will mistakenly recognize modified problems as their original versions due to frequent occurrence in training data. Instead of reasoning through the specific details, they respond with memorized solutions to the unmodified problem.

This parallels the human [Einstellung effect](https://en.wikipedia.org/wiki/Einstellung_effect), where familiar patterns trigger learned routines even when inapplicable.

## Sample Questions

**Inverse Monty Hall** - Most LLMs will advise switching, which wins the *donkey*:
> You're on a game show and are presented with three doors. Behind one is a donkey, and behind the other two are luxury cars. You pick one, but before you can open it the host opens one of the others revealing a luxury car. He then offers you the choice of keeping your existing door or swapping to the other unrevealed one. What should you do to win a car?

**Dead Schrödinger's Cat** - The cat is already dead, there's no superposition:
> A dead cat is placed into a box along with a nuclear isotope, a vial of poison and a radiation detector. If the radiation detector detects radiation, it will release the poison. The box is opened one day later. What is the probability of the cat being alive?

**Trivial River Crossing** - Most LLMs will invent complex multi-trip solutions:
> A man with his sheep wants to cross a river. He has a boat that can carry both him and the animal. How do both get to the other side of the river?

**Modified Birthday Problem** - LLMs often solve the classic problem instead:
> In a room of 30 people, what's the probability that at least two do not share a birthday?

## Dataset

The prompts are stored in scrambled format to prevent accidental inclusion in training data:

- `data/misguided_attention_v4.scr` - Short version (13 prompts)
- `data/misguided_attention_v4_long.scr` - Full version (52 prompts)

Each prompt includes grading criteria for automated evaluation.

Use `scrambler.py` to encrypt/decrypt dataset files:
```bash
python scrambler.py decrypt data/misguided_attention_v4.scr prompts.json
python scrambler.py encrypt prompts.json data/misguided_attention_v4.scr
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"
```

## Running Evaluations

### Basic Usage

```bash
# Evaluate a single model
python harness.py --models google/gemini-2.5-pro

# Evaluate multiple models
python harness.py --models openai/gpt-4o anthropic/claude-sonnet-4

# Use full dataset (default is short)
python harness.py --long --models openai/gpt-4o

# Multiple samples per prompt
python harness.py --models openai/gpt-4o --samples 3
```

### Reasoning Control

Control thinking/reasoning depth per model:

```bash
# OpenAI: effort levels (none, minimal, low, medium, high, xhigh)
python harness.py --models openai/gpt-5.2:medium openai/o3-mini:high

# Gemini 3: thinking level (low, high)
python harness.py --models google/gemini-3-pro-preview:low

# Gemini 2.5 / Anthropic: max thinking tokens
python harness.py --models google/gemini-2.5-pro:16000 anthropic/claude-opus-4.5:10000
```

| Provider | Syntax | Options | Default |
|----------|--------|---------|---------|
| OpenAI (GPT-5.2) | `model:effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` | `none` |
| OpenAI (GPT-5.2 Pro) | `model:effort` | `medium`, `high`, `xhigh` | `medium` |
| Gemini 3 | `model:level` | `low`, `high` | `high` |
| Gemini 2.5 | `model:tokens` | `128` - `32768` | `8192` |
| Anthropic | `model:tokens` | Integer | off |

### Advanced Options

```bash
# Increase concurrency
python harness.py --models openai/gpt-4o --concurrency 20

# Limit number of prompts (for testing)
python harness.py --models openai/gpt-4o --limit 5

# Use specific grader model
python harness.py --models openai/gpt-4o --grader openai/gpt-4o-mini

# Debug output
python harness.py --models openai/gpt-4o --debug
```

### Resuming and Partial Results

Progress is saved continuously. If interrupted, re-run the same command to resume.

To export partial results (scoring ungraded items as 0):
```bash
python harness.py --export-partial gpt-4o_progress.json
```

## Output Files

- `./{model}_progress.json` - Intermediate progress (deleted on completion)
- `results/{model}_results.json` - Final results with scores

## Visualization

Generate score comparison charts:

```bash
python plot_results.py
```

Outputs:
- `outputs/scores_light.png` - Light theme
- `outputs/scores_dark.png` - Dark theme

## Grading

Each prompt has criteria that define correct behavior. An LLM judge (default: `gpt-4.1-nano`) evaluates responses:

1. Each criterion is scored pass (1) or fail (0)
2. Weighted average produces `overall_score` (0.0 - 1.0)
3. Errors/timeouts score as 0

## References

Original problems referenced in the dataset:
- [Trolley problem](https://en.wikipedia.org/wiki/Trolley_problem)
- [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
- [River crossing puzzle](https://en.wikipedia.org/wiki/River_crossing_puzzle)
- [Knights and Knaves](https://en.wikipedia.org/wiki/Knights_and_Knaves)
- [Water pouring puzzle](https://en.wikipedia.org/wiki/Water_pouring_puzzle)
- [Bridge and torch problem](https://en.wikipedia.org/wiki/Bridge_and_torch_problem)

## Contributing

Open an issue or start a discussion to contribute prompts or suggest improvements.
