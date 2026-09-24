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

PROSER is initialized from the selected Vanilla checkpoint, adds five dummy
classifiers, and fine-tunes the complete network for 50 epochs. Its objective
implements classifier placeholders and different-class manifold mixup after
`layer2`, following Zhou, Ye, and Zhan (CVPR 2021):

```bash
python -m task4.train_proser --data-root /path/to/cifar-data
python -m task4.calibrate_proser --data-root /path/to/cifar-data
```

The implementation adapts the loss construction and strongest-dummy handling
from the authors' public reference implementation:
https://github.com/zhoudw-zdw/CVPR21-Proser

After all checkpoints and thresholds are frozen, run the one-time final
evaluation on the official CIFAR-10 test set and the fixed CIFAR-100 Near/Far
groups:

```bash
python -m task4.evaluate_final \
  --cifar10-root /path/to/cifar10 \
  --cifar100-root /path/to/cifar100
```
