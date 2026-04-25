# Adversarial Image Auditor

A multi-modal safety evaluation system designed for Text-to-Image (T2I) generation. This project uses a combination of vision-language alignment, diffusion-aware feature modulation (FiLM), and multi-task learning to detect hidden safety violations and adversarial artifacts in AI-generated images.

## Overview

The Adversarial Image Auditor is a comprehensive tool for detecting safety violations (nudity and violence) and adversarial manipulations. It provides not only classification but also visual heatmaps for interpretability, allowing users to understand why an image was flagged.

## Model Architecture

- **Backbone**: Convolutional CLIP-ResNet 101 backbone for robust feature extraction.
- **Conditioning**: Cross-Attention mechanisms to calculate Prompt Faithfulness by aligning image features with text.
- **Timestep Sensitivity**: Integrates Timestep-aware FiLM conditioning to adjust detection sensitivity based on the diffusion generation stage.
- **Multi-Task Heads**: Specialized heads for adversarial scoring, category-wise safety classification (Safe, Nudity, Violence), and seam quality assessment (via LPIPS).
- **Explainability**: GradCAM is used to pull activations from the last conv layer of the ResNet backbone. These are bilinearly extrapolated and thresholded to create a binary map for targeted inpainting and verification.

For a detailed technical breakdown, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Performance Summary

Based on the final 3-class evaluation:

| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Safe | 0.90 | 0.97 | 0.94 |
| Nudity | 0.90 | 0.78 | 0.84 |
| Violence | 0.85 | 0.89 | 0.87 |
| **Average** | **0.88** | **0.88** | **0.88** |

**Overall Accuracy**: 88.36%

For more metrics, see [SUMMARY.md](SUMMARY.md).

## Installation

Ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

To train or evaluate the model using the final optimized script:

```bash
python train_final.py
```

## Credits

Developed by Paras Dhiman.
