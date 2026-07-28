# PdrQC: Pauli-Space Discriminative Quantum Classification

This directory is the compact code package for the PdrQC paper. Executable source
files, one root-level notebook, a small processed data set, dependency information,
and an ignored output directory are kept together.

The included data and default settings are intended only for a quick execution check.
They do not reproduce the full-data numerical values reported in the paper.

## Contents

- `pauli_upload_analysis.py`: MNIST/Fashion-MNIST preprocessing, standardized PCA
  angle mapping, atomic upload blocks, state-vector simulation, Pauli features,
  qubit-wise commuting grouping, and finite-shot sampling.
- `pauli_progressive_selector.py`: the PdrQC progressive search. Each search round
  appends one atomic block, evaluates low-weight Pauli features, selects a compact
  multiclass-discriminative Pauli subspace, and freezes the final classifier.
- `pdrqc_paper_experiments.py`: the comparisons explicitly used in the paper:
  PCA-LR, Full Pauli, computational-basis readout, the data re-uploading classifier
  (DRC), and the quantum-kernel classifier (QKC).
- `01_pdrqc_tiny_mnist_demo.ipynb`: a top-to-bottom small-sample execution check.
- `dataset/`: a tiny processed MNIST subset and an optional downloader for the full
  MNIST/Fashion-MNIST data.
- `results/`: default output directory for fresh runs; generated JSON files are
  ignored by Git.

Only methods and evaluation mechanisms described in the paper are included.

## Provided Data

The package contains only a small balanced MNIST subset:

- Training archive: `dataset/mnist/processed/train.csv.gz`
- Test archive: `dataset/mnist/processed/test.csv.gz`
- Training size: 200 images, 20 per digit
- Test size: 100 images, 10 per digit
- Features: 784 pixels and one `label` column

The default demo uses digits 0, 1, and 2, four qubits, 20 available training rows per
class, five validation rows per class, and ten test rows per class.

## Environment

```bash
pip install -r requirements.txt
```

The implementation uses local exact-state simulation. The finite-shot evaluation
models sampling uncertainty only; it does not model gate noise, decoherence, crosstalk,
or biased readout.

## Quick PdrQC Run

```bash
python pauli_progressive_selector.py \
  --dataset mnist \
  --classes 0,1,2 \
  --n-qubits 4 \
  --max-pauli-weight 2 \
  --max-rounds 3 \
  --train-per-class 20 \
  --validation-per-class 5 \
  --test-per-class 10 \
  --shot-repeats 5
```

The output is written to `results/pdrqc_tiny_mnist.json`.

## Run the Paper Classifiers on the Tiny Data

```bash
python pdrqc_paper_experiments.py \
  --dataset mnist \
  --classes 0,1,2 \
  --n-qubits 4 \
  --max-rounds 3 \
  --drc-epochs 20 \
  --shot-repeats 5
```

This command runs PdrQC and only the comparison methods present in the paper. For QKC,
the reported inference cost is

```text
number of support vectors × shots per kernel value.
```

In contrast, the PdrQC shot budget is the total per-sample budget distributed among
the selected commuting Pauli groups.

## Full MNIST and Fashion-MNIST Data

The full processed data are intentionally not included. Download and convert the public
IDX files with:

```bash
python dataset/download_and_prepare_image_datasets.py \
  --datasets mnist,fashion-mnist \
  --output dataset \
  --overwrite
```

After full data are prepared, use eight qubits and the larger per-class sample caps
described in the paper. The small notebook remains a smoke test and should not be used
as evidence for the paper's reported accuracy.
