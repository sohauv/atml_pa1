# ATML Programming Assignment 1

Code, experiment configurations, machine-readable results, and reproducibility
instructions for Programming Assignment 1 in Advanced Topics in Machine
Learning (Fall 2026).

## Repository status

| Task | Topic | Status | Instructions |
|---|---|---:|---|
| 1 | Inductive Biases and Feature Representations | Complete | [`task1/README.md`](task1/README.md) |
| 2 | Unsupervised Domain Adaptation | Pending | To be added |
| 3 | Domain Generalization | Pending | To be added |
| 4 | Open-Set Recognition | Pending | To be added |

## Setup

Clone the repository together with its external submodules:

```bash
git clone --recurse-submodules https://github.com/sohauv/atml_pa1.git
cd atml_pa1
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the environment dependencies:

```bash
python -m pip install -r requirements.txt
```

Detailed execution commands, configurations, manual review steps, and output
locations are documented in each task's README.

## Reproducibility conventions

- Random seed `6304` is used where specified by the assignment.
- Experiment settings are stored under each task's `configs/` directory.
- Dataset split indices or identifiers are saved with the task results.
- Small JSON/CSV results and report-supporting figures are committed.
- Raw datasets, feature caches, generated datasets, downloaded weights, and
  large checkpoints are excluded from Git.
- Task boundaries are kept explicit; only genuinely shared utilities belong in
  `common/`.

## Current layout

```text
atml_pa1/
  README.md
  requirements.txt
  common/
  task1/
    README.md
    configs/
    data/
    models/
    analysis/
    scripts/
    results/
```

## External implementations

Materially reused external implementations are identified in the relevant
task README. Task 1 uses
[`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) as a
Git submodule for cue-conflict generation.
