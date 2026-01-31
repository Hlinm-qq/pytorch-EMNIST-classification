# EMNIST Classification with PyTorch

This repository provides a full training and evaluation pipeline for the [**EMNIST dataset**](https://www.nist.gov/itl/products-and-services/emnist-dataset) — a large-scale handwritten character dataset that extends MNIST to include both digits and letters.

---

## Features

* EMNIST splits:
  * `balanced`
  * `byclass`
  * `bymerge`
  * `digits`
  * `letters`
  * `mnist`

* Command-line argument control (`--mode`, `--split`, etc.)
* Automatic device selection (CUDA or CPU)
* Built-in visualization and evaluation tools

---

## Requirements

Before running, install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. **Train a Model**

To train a model from scratch on a specific EMNIST split:

```bash
python main.py --mode train --split byclass
```

* Downloads EMNIST if not found.
* Splits training set into train/validation.
* Trains a CNN model.
* Saves the best-performing model as `best_emnist_model_base.pth`.
* Generates plots:

  * `emnist_training_history.png`
  * `emnist_inference.png`
  * `emnist_sample_visualization.png`

---

### 2. **Evaluate a Saved Model**

After training, evaluate on the test set:

```bash
python main.py --mode eval --split byclass
```

* Loads `best_emnist_model_base.pth`.
* Runs evaluation metrics.
* Saves inference visualizations.

---

## Key Components

### `EMNISTDataset`

A wrapper around `torchvision.datasets.EMNIST` providing:

* Default data augmentation
* Normalization
* Compatibility with `DataLoader`
* Dataset download and split handling

### `CustomEMNISTCNN`

A simple CNN model tailored for EMNIST.
Defined in `custom_network/model.py` — you can modify it freely to test different architectures.

### `EMNISTTrainer`

Handles:

* Training and validation loops
* Early stopping (via `PATIENCE`)
* Learning rate configuration
* Saving/loading the best model

### `evaluation/tools.py`

Includes:

* Accuracy, precision, recall, and F1-score computation
* Inference visualizations

---

## Command-Line Arguments

| Argument                                                 | Type    | Default   | Description                                |
| -------------------------------------------------------- | ------- | --------- | ------------------------------------------ |
| `--mode`                                                 | `str`   | `train`   | Choose between `train` or `eval` |
| `--split`                                                | `str`   | `byclass` | EMNIST split to use                        |

---

## Training Configuration

You can customize hyperparameters directly in the script:

```python
BATCH_SIZE = 400
N_EPOCHS = 10
PATIENCE = 7
LEARNING_RATE = 0.001
VAL_RATIO = 0.1
SEED = 42
```

---

## Notes

* Mapping files in `/mapping/` are required for character label decoding.
* EMNIST dataset will be automatically downloaded into the specified root directory.
* All output artifacts are saved in the working directory.

---

## Citation

```
@article{cohen2017emnist,
  title={EMNIST: an extension of MNIST to handwritten letters},
  author={Cohen, Gregory and Afshar, Saeed and Tapson, Jonathan and van Schaik, André},
  journal={arXiv preprint arXiv:1702.05373},
  year={2017}
}
```
