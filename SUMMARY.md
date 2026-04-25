### Model Architecture 

- Uses a convolution-based **CLIP-ResNet 101** backbone for robust feature extraction. 
- Employs **Cross-Attention** to calculate **Prompt Faithfulness** by aligning image features with text embeddings.
- One head outputs a unified **adversarial score** for the image.
- One head outputs **category-wise probabilities** (e.g., Safe, Nudity, Violence).
- Integrates **Timestep-aware FiLM conditioning** to adapt detection sensitivity across different diffusion generation stages.
- Uses **LPIPS** to assess **seam quality**, identifying unnatural inpainting/composition artifacts.
- Produces a localized **adversarial heatmap** over the image.
- Uses **GradCAM** during inference to extract activations from the final ResNet layer; these are bilinearly extrapolated and thresholded to create a **binary map** for targeted inpainting.

---

**Credits:** Paras Dhiman 

### Auditor Model Performance

| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Safe** | 0.90 | 0.97 | 0.94 |
| **Nudity** | 0.90 | 0.78 | 0.84 |
| **Violence** | 0.85 | 0.89 | 0.87 |
| **Average** | **0.88** | **0.88** | **0.88** |

**Overall Accuracy:** 88.36%
