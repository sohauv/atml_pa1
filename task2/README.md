# Task 2: Unsupervised Domain Adaptation on PACS

This directory implements and evaluates Source-only, DAN, DANN, and CDAN on
PACS. The labelled source domains are Photo, Art Painting, and Cartoon. Sketch
is the unlabeled adaptation domain and the final evaluation domain.

## Protocol

- Seed: `6304`
- Source split: per-domain, class-stratified 80/20 train/validation split
- Source domains: `photo`, `art_painting`, `cartoon`
- Target domain: `sketch`
- Backbone: torchvision ResNet-18 with `IMAGENET1K_V1` weights
- Classifier: 512-dimensional ResNet feature to 7 PACS classes
- Training transform: resize to 256, random 224 crop, random horizontal flip,
  ImageNet normalization
- Evaluation transform: resize to 256, center 224 crop, ImageNet normalization
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`
- Maximum epochs: 30
- Early-stopping patience: 5
- Checkpoint selection: mean macro-F1 over the three source validation splits
- Batch composition: 8 images from each source domain (24 source images total)
  and 24 unlabeled Sketch images for adaptation methods
- Batch-normalization running means and variances remain fixed at their
  ImageNet values; affine scale and bias parameters remain trainable

The fixed split is stored in
`../shared/splits/pacs_sketch_seed6304.json` and is reused by every Task 2 run
and by Task 3. Sketch labels are not used during adaptation or checkpoint
selection. They are read only by `evaluate_final.py`, after all methods,
hyperparameters, and checkpoints have been fixed.

## Methods

### Source-only

Fine-tunes the complete ResNet-18 and classifier using labelled source data
only. This checkpoint is also the unchanged ERM baseline for Task 3.

### DAN

Adds multi-kernel maximum mean discrepancy (MMD) between source and target
features. The three RBF bandwidths are `0.5`, `1`, and `2` times the current
batch median pairwise squared distance. The main run uses MMD weight
`lambda=1`.

### DANN

Uses a `512 -> 256 -> 2` domain discriminator with ReLU, dropout `0.5`, and a
scheduled gradient-reversal layer.

### CDAN

Uses the outer product between the 512-dimensional feature vector and the
7-class probability vector, giving a 3,584-dimensional discriminator input.
Neither features nor probabilities are detached, and entropy conditioning is
not used.

For numerical stability, DANN and CDAN run in float32, use global gradient-norm
clipping at 5, and L2-normalize features only on the domain-discriminator path.
The classifier continues to receive the original unnormalized features. These
settings were fixed before final Sketch-label evaluation.

## Dataset setup

The PACS release used here has the structure:

```text
pacs/
  images/
    photo/
    art_painting/
    cartoon/
    sketch/
```

On Kaggle, it can be downloaded and extracted with:

```python
from pathlib import Path

pacs_root = Path("/kaggle/working/datasets/pacs_extracted/pacs")

if not pacs_root.exists():
    !pip install -q gdown
    !mkdir -p /kaggle/working/datasets
    !gdown 1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE \
        -O /kaggle/working/datasets/pacs.zip
    !unzip -q /kaggle/working/datasets/pacs.zip \
        -d /kaggle/working/datasets/pacs_extracted
```

The expected image counts are:

| Domain | Images |
|---|---:|
| Photo | 1,670 |
| Art Painting | 2,048 |
| Cartoon | 2,344 |
| Sketch | 3,929 |
| Total | 9,991 |

## Training commands

Create the fixed split and train Source-only once:

```bash
python -m task2.train \
  --method source_only \
  --data-root /path/to/pacs \
  --run-name source_only \
  --create-split
```

Reuse that split for every subsequent run:

```bash
python -m task2.train --method dan  --data-root /path/to/pacs --run-name dan --mmd-weight 1.0
python -m task2.train --method dann --data-root /path/to/pacs --run-name dann
python -m task2.train --method cdan --data-root /path/to/pacs --run-name cdan
```

Controlled DAN study:

```bash
python -m task2.train --method dan --data-root /path/to/pacs --run-name dan_lambda0_1 --mmd-weight 0.1
python -m task2.train --method dan --data-root /path/to/pacs --run-name dan_lambda10 --mmd-weight 10.0
```

## Final evaluation

Run this only after all checkpoints and settings are fixed:

```bash
python -m task2.evaluate_final \
  --data-root /path/to/pacs \
  --run source_only=task2/checkpoints/source_only_best.pt \
  --run dan=task2/checkpoints/dan_best.pt \
  --run dann=task2/checkpoints/dann_best.pt \
  --run cdan=task2/checkpoints/cdan_best.pt \
  --run dan_lambda0_1=task2/checkpoints/dan_lambda0_1_best.pt \
  --run dan_lambda10=task2/checkpoints/dan_lambda10_best.pt \
  --output-directory task2/results/final
```

Create reporting artifacts from the locked evaluation outputs:

```bash
python -m task2.evaluation.plot_task2_results \
  --results-dir task2/results/final \
  --data-root /path/to/pacs \
  --output-dir task2/results/reporting
```

## Final results

| Run | Selected epoch | Mean source-val macro-F1 | Sketch accuracy | Sketch macro-F1 | Domain separability |
|---|---:|---:|---:|---:|---:|
| Source-only | 5 | 0.9401 | 0.6829 | 0.6727 | 0.9973 |
| DAN, lambda=1 | 11 | 0.9567 | 0.6691 | 0.6319 | 0.7940 |
| DANN | 6 | 0.9411 | 0.3706 | 0.4622 | 0.9945 |
| CDAN | 6 | 0.9409 | 0.2105 | 0.0705 | 0.9986 |
| DAN, lambda=0.1 | 3 | 0.9352 | 0.6299 | 0.6157 | 0.9863 |
| DAN, lambda=10 | 1 | 0.0507 | 0.0407 | 0.0112 | 0.6799 |

Domain separability uses equal numbers of source-validation and target features,
a seeded 70/30 split, and balanced logistic regression with `C=1`.

## Output inventory

- `results/configs/`: resolved configuration for every fixed run
- `results/training/`: per-epoch losses and source-validation metrics
- `results/final/final_summary.csv`: aggregate comparison table
- `results/final/final_results.json`: complete evaluation data
- `results/final/predictions/`: per-image Sketch predictions
- `results/reporting/main_method_comparison.png`: target scores and separability
- `results/reporting/dan_lambda_study.png`: controlled alignment-strength study
- `results/reporting/confusion_matrices.png`: normalized target confusion matrices
- `results/reporting/per_class_f1.png`: per-class target comparison
- `results/reporting/target_failure_examples.png`: deterministic failure examples

Model checkpoints and PACS images are intentionally excluded from Git.
