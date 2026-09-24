# Task 4: Open-set recognition

Task 4 uses CIFAR-10 only for training, checkpoint selection, score design, and
threshold calibration. CIFAR-100 is evaluation-only and must not be loaded by
the training program.

Vanilla and GCSC use the same CIFAR ResNet-18, SGD schedule, split, and seed.
GCSC changes only the training transform by adding
`RandAugment(num_ops=2, magnitude=9)`.

```bash
python -m task4.train --method vanilla --data-root datasets --create-split
python -m task4.train --method gcsc --data-root datasets
```

Both runs train for 100 epochs and retain the checkpoint with the highest
CIFAR-10 validation accuracy.

After both runs, freeze their checkpoints and calibrate the required open-set
scores using only CIFAR-10 training and validation data:

```bash
python -m task4.calibrate_scores --data-root /path/to/cifar-data
```

The command fits Vanilla Mahalanobis statistics on the unaugmented CIFAR-10
training split and fixes every rejection threshold at the 95th percentile of
the CIFAR-10 validation unknownness scores. It does not evaluate CIFAR-10 test
or load CIFAR-100.
