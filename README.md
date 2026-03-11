# MRI Super Resolution

This repository contains the core code used for our MRI super-resolution project, which studies whether low-field **64 mT** brain MRI scans can be enhanced to better approximate paired high-field **3 T** scans using a residual U-Net-based deep learning model.

## Overview

Our approach uses a hybrid volumetric super-resolution model built around a lightweight **3D U-Net** backbone enhanced with:

- **RRDB-based feature extraction**
- **3D CBAM attention**
- **anisotropic upsampling** for depth and in-plane resolution enhancement

The final submitted model was a **CNN-only version** of the architecture, rather than the earlier GAN-based setup explored during tuning.

## Competition

This project was developed for the [MRI Super Resolution Challenge Part 2 - CS-GY 9223](https://www.kaggle.com/competitions/mri-super-resolution-challenge-part-2-cs-gy-9223) Kaggle competition and achieved **2nd place overall**.

## Repository Contents

- `rrdbunetpatch.py` — main model definition for the residual U-Net / RRDB-based super-resolution network
- `test.py` — inference / testing script
- `extract_slices.py` — preprocessing helper for extracting slices from MRI volumes
- `metric.py` — competition evaluation metric
- `.gitignore`

## Notes

The paper mentions a Bayesian optimization script (`bayes_opt.py`) used during hyperparameter tuning, but that file is **not included** in this repository. This repo contains the final model, testing/inference code, and the provided competition utility scripts.

## Method Summary

The model is based on a lightweight 3D U-Net with one downsampling and one upsampling stage. Standard feature blocks were replaced with **RRDB-enhanced blocks** to improve feature reuse and stability.

After feature extraction, the model applies a **two-stage anisotropic upsampling strategy**:

1. **3D trilinear interpolation** to increase depth
2. Reshaping into 2D slices and applying **bicubic interpolation** per slice for in-plane upsampling

A **3D CBAM module** is then used to refine features before final output projection.

## Training Context

During development, hyperparameters were tuned on an earlier GAN-based variant, but the final best-performing submission used a reduced-width **CNN-only generator** trained with the following loss:

$$
L_G = 0.5 \cdot L_{L1} + 0.5 \cdot (1 - \text{MS-SSIM})
$$

The final model was trained on the full dataset for 100 epochs.

## Results

The final submission achieved:

- **Public leaderboard MS-SSIM:** `0.6419`
- **Private leaderboard MS-SSIM:** `0.6473`

## Usage

Since this repo only includes the core model and competition utilities, the typical workflow is:

1. Preprocess or extract slices using `extract_slices.py`
2. Run inference with `test.py`
3. Evaluate predictions with `metric.py`

You may need to adapt file paths and dataset locations depending on your local setup.

## Acknowledgments

This work builds on ideas from U-Net, ESRGAN/RRDB, and CBAM, adapted here for volumetric MRI super-resolution under limited-data conditions.
