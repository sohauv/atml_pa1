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
