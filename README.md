# ATML Programming Assignment 1

Code, experiment configurations, machine-readable results, and reproducibility
instructions for Programming Assignment 1 in Advanced Topics in Machine
Learning (Fall 2026).

## Repository status

| Task | Topic | Status | Instructions |
|---|---|---:|---|
| 1 | Inductive Biases and Feature Representations | Complete | [`task1/README.md`](task1/README.md) |
| 2 | Unsupervised Domain Adaptation | Complete | [`task2/README.md`](task2/README.md) |
| 3 | Domain Generalization | Complete | [`task3/README.md`](task3/README.md) |
| 4 | Open-Set Recognition | Complete | [`task4/README.md`](task4/README.md) |

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

## Reproducing the tasks

Run each task from the repository root. Dataset locations and complete commands
are documented in the linked task README files.

1. **Task 1:** train the three frozen-feature linear probes, evaluate clean and
   intervention sets, manually review the generated cue conflicts, and then run
   the representation and aggregation scripts.
2. **Task 2:** create the shared PACS source split once, train Source-only, DAN,
   DANN, CDAN, and the fixed DAN weight study, then run final evaluation and
   reporting only after all checkpoints are frozen.
3. **Task 3:** reuse Task 2's source-only checkpoint and PACS split, train
   DAN-DG and SAM without loading Sketch, run source-only diagnostics, and only
   then perform the one-time final Sketch evaluation and reporting.
4. **Task 4:** train Vanilla and GCSC, calibrate their scores using CIFAR-10
   only, train and calibrate PROSER, and finally evaluate the frozen methods on
   CIFAR-10 test and the fixed CIFAR-100 Near/Far groups.

## Reproducibility conventions

- Random seed `6304` is used where specified by the assignment.
- Experiment settings are stored under each task's `configs/` directory.
- Dataset split indices or identifiers are saved with the task results.
- Small JSON/CSV results and report-supporting figures are committed.
- Raw datasets, feature caches, generated datasets, downloaded weights, and
  large checkpoints are excluded from Git.
- Task boundaries are kept explicit; PACS utilities shared by Tasks 2 and 3
  belong in `shared/`, while Task 1's general seed helper remains in `common/`.

## Current layout

```text
atml_pa1/
  README.md
  requirements.txt
  common/
  shared/
  task1/
    README.md
    configs/
    data/
    models/
    analysis/
    scripts/
    results/
  task2/
    README.md
    configs/
    methods/
    models/
    evaluation/
    results/
  task3/
    README.md
    configs/
    methods/
    evaluation/
    results/
  task4/
    README.md
    configs/
    data/
    methods/
    models/
    evaluation/
    results/
```

## External implementations

Materially reused external implementations are identified in the relevant
task README. Task 1 uses
[`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) as a
Git submodule for cue-conflict generation. Task 4's PROSER objective adapts
the loss construction and strongest-dummy handling from the authors' public
reference implementation, which is linked in `task4/README.md`.
