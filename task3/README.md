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

All model selection uses mean source-validation macro-F1. The three DAN-DG
weights are therefore 0.1, 1, and 10, with `dan_dg` representing weight 1.

Every update draws exactly eight examples from each source domain. Incomplete
final batches are dropped with `drop_last=True`, and the smaller domain loaders
are cycled. DAN-DG and SAM run in float32 and use global gradient-norm clipping
at 5 as numerical-stability measures. For DAN-DG, features entering the MMD
term are L2-normalized while the classifier continues to receive the original
unnormalized features. This additional numerical-stability modification was
introduced because stronger MMD weights caused feature-scale collapse and
failure to learn; it keeps the alignment term finite without changing the
classification pathway. These settings use only source training and validation
behavior, do not use Sketch data, and are fixed before source-only diagnostics
or final Sketch evaluation.

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
ascent sharpness proxy. For the domain probe, each feature dimension is
standardized using statistics fitted only on the probe-training split before
balanced logistic regression; this avoids solver failures caused by
ill-conditioned feature scales without using the held-out probe split or
Sketch data. Outputs are written to `task3/results/source_diagnostics/`.

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

`training_curves.png` separately reports classification loss, DAN-DG MMD
alignment loss, the total objective, and the source-validation selection
metric.

## Fixed results

| Run | Mean source macro-F1 | Source separability | Sharpness increase | Sketch accuracy | Sketch macro-F1 |
|---|---:|---:|---:|---:|---:|
| ERM | 0.9401 | 0.8505 | 0.2255 | 0.6829 | 0.6727 |
| DAN-DG, lambda 0.1 | 0.9402 | 0.7542 | 1.6264 | 0.6918 | 0.7138 |
| DAN-DG, lambda 1 | 0.9081 | 0.6213 | 83.4487 | 0.5548 | 0.4479 |
| DAN-DG, lambda 10 | 0.0507 | 0.6678 | 6.6424 | 0.0407 | 0.0112 |
| SAM, rho 0.05 | 0.9552 | 0.8405 | 0.1618 | 0.7200 | 0.7528 |

Source-domain separability has chance level `1/3`. Increasing the DAN-DG weight
reduces recoverable source-domain information, but this does not reliably
preserve class information: weight 1 substantially damages source and Sketch
recognition, while weight 10 collapses classification. Weight 0.1 retains
source performance and modestly improves Sketch over ERM. SAM gives the best
Sketch accuracy and macro-F1 and the lowest measured sharpness increase among
the non-collapsed models.
