# Cell Counting in Microscopy Images with U-Net

## Overview

This project trains a U-Net convolutional neural network to segment and count cells in 128x128 grayscale microscopy images. The model learns to produce binary segmentation masks that identify cell regions, then uses connected-component analysis to count individual cells in each image.

The project was developed for MSDS 373 (Deep Learning) and demonstrates end-to-end biomedical image segmentation: data loading, model training, qualitative evaluation, and automated cell counting on held-out test images.

## Architecture

The model follows the standard U-Net encoder-decoder structure ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)):

- **Encoder:** Four downsampling stages (64 → 128 → 256 → 512 channels), each with two 3x3 convolutions + ReLU followed by 2x2 max-pooling.
- **Bottleneck:** Two 3x3 convolutions at 1024 channels.
- **Decoder:** Four upsampling stages that mirror the encoder. Each stage upsamples, concatenates the corresponding encoder features via skip connections, then applies two 3x3 convolutions.
- **Output:** A final convolution produces a single-channel segmentation map (logits), trained with binary cross-entropy loss.

## Results

| Metric | Value |
|---|---|
| Epochs | 65 |
| Learning Rate | 0.0003 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |

![Training Curves](images/training_curves.png)

![Segmentation Results](images/segmentation_results.png)

> Save these images from the notebook after training to populate the figures above.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the dataset files and place them in the `data/` directory:

```
data/
├── train_data.npz    (contains 'X' and 'y' arrays)
└── test_images.npz   (contains 'X' array)
```

The `.npz` files contain NumPy arrays of 128x128 grayscale microscopy images. `train_data.npz` includes both input images (`X`) and binary segmentation masks (`y`). `test_images.npz` contains only input images for evaluation.

## Usage

Open and run the notebook:

```bash
jupyter notebook unet_cell_counting.ipynb
```

The notebook will:
1. Load and split the training data (80/20 train/validation)
2. Train the U-Net for 65 epochs
3. Plot training and validation loss curves
4. Visualize predicted segmentation masks alongside ground truth
5. Count cells in the test set and save results to `cell_counts.csv`

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
