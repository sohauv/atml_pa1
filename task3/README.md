# Task 3: Domain generalization on PACS

Task 3 trains exclusively on the PACS source domains `photo`, `art_painting`,
and `cartoon`. The target domain `sketch` is not loaded by `task3.train` and
must not be used for checkpoint selection, hyperparameter selection, or source
diagnostics.

The Task 2 source-only checkpoint is reused as the ERM baseline. Do not retrain
that baseline with different settings.

## Required state

- PACS extracted so its root resolves to the directory above `images/` or to
  the `images/` directory itself.
- `shared/splits/pacs_sketch_seed6304.json` from Tasks 2 and 3.
- `task2/checkpoints/source_only_best.pt` restored from the private Kaggle
  state dataset.

## Source-only training

Main DAN-DG model:

```bash
python -m task3.train \
  --method dan_dg \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs
```

Main SAM model:

```bash
python -m task3.train \
  --method sam \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs
```

Controlled DAN-DG alignment study, fixed before any Task 3 Sketch evaluation:

```bash
python -m task3.train --method dan_dg \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs \
  --alignment-weight 0.1 --run-name dan_dg_lambda0_1

python -m task3.train --method dan_dg \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs \
  --alignment-weight 10 --run-name dan_dg_lambda10
```

All model selection uses mean source-validation macro-F1. The four DAN-DG
weights are therefore 0.1, 1, and 10, with `dan_dg` representing weight 1.

## Protocol guardrail

Do not run a Sketch evaluation until ERM, DAN-DG, SAM, the controlled-study
settings, source-domain separability, and sharpness-proxy decisions are fixed.

## Source-only diagnostics

After every checkpoint and controlled-study setting has been fixed, run the
source-validation diagnostics. This stage still does not load Sketch:

```bash
python -m task3.evaluate_sources \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs
```

This produces per-source accuracy and macro-F1, mean and worst-source scores,
a balanced source-domain linear-probe accuracy, and the fixed-batch normalized
ascent sharpness proxy. Outputs are written to
`task3/results/source_diagnostics/`.

## Final Sketch evaluation

Only after the source-only stage is complete, run the one-time final target
evaluation:

```bash
python -m task3.evaluate_final \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs
```

The evaluator records aggregate and per-class Sketch metrics, confusion
matrices, prediction CSVs, and changes relative to the reused ERM baseline in
`task3/results/final/`.

## Reporting artifacts

```bash
python -m task3.evaluation.plot_task3_results \
  --data-root /kaggle/working/datasets/pacs_extracted/pacs
```

Figures and their compact source tables are written to
`task3/results/reporting/`.

## Fixed results

| Run | Mean source macro-F1 | Source separability | Sharpness increase | Sketch accuracy | Sketch macro-F1 |
|---|---:|---:|---:|---:|---:|
| ERM | 0.9401 | 0.8738 | 0.2255 | 0.6829 | 0.6727 |
| DAN-DG, lambda 0.1 | 0.9483 | 0.7442 | 0.1671 | 0.6931 | 0.7182 |
| DAN-DG, lambda 1 | 0.0507 | 0.3322 | 0.0138 | 0.0407 | 0.0112 |
| DAN-DG, lambda 10 | 0.0507 | 0.3322 | 0.0484 | 0.0407 | 0.0112 |
| SAM, rho 0.05 | 0.9576 | 0.8571 | 0.1129 | 0.6434 | 0.6805 |

Source-domain separability has chance level `1/3`. The near-chance values for
DAN-DG weights 1 and 10 accompany chance-level classification, showing feature
collapse rather than useful domain invariance. The controlled weight 0.1
reduces separability while retaining class information and gives the strongest
Sketch macro-F1.
