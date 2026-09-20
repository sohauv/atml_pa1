# Task 1: Inductive Biases and Feature Representations

This directory contains the code and saved results for Task 1 of ATML PA1. The
experiments compare frozen ResNet-50, ViT-B/16, and CLIP ViT-B/32 backbones on
STL-10 under controlled color, shape/texture, translation, and patch-structure
interventions.

## Experimental setup

- Dataset: STL-10
- Split seed: `6304`
- Official training partition: stratified 80/20 train/validation split
- Evaluation set: class-balanced subset of 500 official test images
- Input size: 224 x 224 RGB before model-specific normalization
- Frozen backbones:
  - torchvision ResNet-50, `ResNet50_Weights.IMAGENET1K_V2`
  - torchvision ViT-B/16, `ViT_B_16_Weights.IMAGENET1K_V1`
  - OpenCLIP ViT-B-32, pretrained weights `openai`
- Classifiers: one learned linear head per frozen backbone
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Maximum epochs: 50
- Early stopping patience: 5 epochs without improved validation accuracy
- CLIP zero-shot prompt: `a photo of a {class}.`

The complete experiment configuration is stored in
[`configs/task1.yaml`](configs/task1.yaml). Exact split identifiers are stored
in [`results/stl10_splits.json`](results/stl10_splits.json).

## Environment setup

Clone the repository with its AdaIN submodule:

```bash
git clone --recurse-submodules https://github.com/sohauv/atml_pa1.git
cd atml_pa1
```

If the repository was cloned without submodules, initialize them separately:

```bash
git submodule update --init --recursive
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU execution is recommended for feature extraction, but the summary plotting
scripts only read saved JSON/CSV files and can run on CPU.

## Reproduction workflow

All commands below are run from the repository root.

### 1. Clean baselines

```bash
python -m task1.scripts.run_task1
```

This command constructs the fixed STL-10 splits, extracts frozen features,
trains the three linear heads, and evaluates the learned heads and CLIP
zero-shot classifier on clean images.

Main outputs:

- `task1/results/stl10_splits.json`
- `task1/results/clean_baselines.json`
- `task1/results/training/`
- `task1/results/predictions/*_clean.csv`

### 2. Color, translation, and patch interventions

```bash
python -m task1.scripts.evaluate_interventions
```

The interventions are:

- grayscale;
- fixed hue rotation with factor `0.5`;
- translations of 8, 16, and 32 pixels in four cardinal directions using
  reflection padding followed by a shifted crop;
- one seeded, non-identity 4 x 4 patch-grid permutation per image.

All models receive the same transformed images. The clean condition is treated
as displacement zero for the translation curves.

Main outputs:

- `task1/results/intervention_results.json`
- `task1/results/predictions/`

### 3. Generate and review cue conflicts

Cue conflicts are produced with AdaIN at style strength `0.8`:

```bash
python -m task1.data.make_cue_conflicts
```

The generation design uses five unordered class pairs in both directions:

- airplane / bird
- car / horse
- deer / ship
- cat / truck
- dog / monkey

Generation is kept separate from evaluation so that every model receives the
same accepted images. Candidate metadata is written to:

```text
task1/results/cue_conflicts/candidates.csv
```

Create contact sheets for manual inspection:

```bash
python -m task1.analysis.make_cue_conflict_sheets
```

Before running any model on the cue conflicts, inspect every candidate and fill
the `accepted` and `rejection_reason` columns. Reject an image when severe
artifacts obscure most of the image or when the main content shape is no longer
visually recognizable. Do not use model predictions during this filtering
step.

The saved reviewed metadata contains 300 generated candidates: 284 accepted
and 16 rejected.

### 4. Evaluate accepted cue conflicts

```bash
python -m task1.scripts.evaluate_cue_conflicts
```

Each prediction is categorized as `shape`, `texture`, or `other`. Shape bias
and coverage are computed as:

```text
shape bias = N_shape / (N_shape + N_texture)
coverage   = (N_shape + N_texture) / N_total
```

Main outputs:

- `task1/results/cue_conflict_results.json`
- `task1/results/cue_conflicts/predictions/`

### 5. Representation analysis

```bash
python -m task1.scripts.run_representation_analysis
```

This measures paired cosine stability for grayscale, translations, 4 x 4 patch
shuffling, and accepted cue conflicts. It also jointly fits UMAP to clean and
transformed features for each backbone and intervention.

UMAP settings:

- neighbors: 15
- minimum distance: 0.1
- metric: cosine
- seed: 6304
- clean and transformed features fitted jointly

Main outputs:

- `task1/results/representation_results.json`
- `task1/results/representation_tables/`
- `task1/results/figures/representations/`

### 6. Aggregate figures and tables

```bash
python -m task1.analysis.plot_task1_results
```

This creates compact performance, translation, cue-conflict, and
representation-stability figures from the saved machine-readable results.

```bash
python -m task1.analysis.make_cue_conflict_examples
```

This deterministically selects cue-conflict agreements, disagreements,
other-class responses, and rejected generation failures. Selection metadata is
saved beside the aggregate result tables.

Main outputs:

- `task1/results/figures/summary/`
- `task1/results/figures/cue_conflict_examples/`
- `task1/results/summary_tables/`

## Directory guide

```text
task1/
  configs/       experiment configuration
  data/          STL-10 splits, datasets, transforms, cue-conflict generation
  models/        frozen backbone wrappers and linear classifier head
  analysis/      metrics, representation analysis, figures, and evidence sheets
  scripts/       experiment entry points
  results/       committed machine-readable results and figures
  cache/         generated features; not committed
  checkpoints/   trained heads; not committed
```

Raw datasets, downloaded model weights, generated cue-conflict images, feature
caches, and checkpoints are intentionally excluded from Git.

## External code and attribution

Cue-conflict generation uses the public
[`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN)
implementation as a Git submodule. The local integration adapts its encoder,
decoder, and adaptive instance-normalization operation to the fixed STL-10
generation workflow. Refer to the upstream repository and Huang and Belongie
(2017) for the original implementation and method.

Pretrained models are loaded through torchvision and OpenCLIP. No pretrained
backbone is fine-tuned in Task 1.
