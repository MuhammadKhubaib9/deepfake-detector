"""Grad-CAM explainability engine (FR-17, Section 5.4.4).

Heatmap generated from the CNN's final convolutional block (Xception conv4),
blended over the analyzed face crop; returned as a PNG overlay.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _overlay(heatmap: np.ndarray, canvas_rgb: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
    """Blend a 0..1 heatmap onto a 0..255 RGB image (jet colormap)."""
    heat = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    jet = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(canvas_rgb, 1.0 - alpha, jet, alpha, 0)


def compute_gradcam_heatmap(cnn_model, face_tensor: torch.Tensor,
                            device: torch.device) -> np.ndarray:
    """0..1 saliency map for a [1, 3, 224, 224] ImageNet-normalised tensor."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layer = cnn_model.grad_cam_target_layer
    cam = GradCAM(model=cnn_model, target_layers=[target_layer])
    grayscale = cam(input_tensor=face_tensor.to(device),
                    targets=[ClassifierOutputTarget(0)])
    return grayscale[0]


def save_heatmap_overlay(saliency: np.ndarray, face_rgb: np.ndarray,
                         out_png: str | Path, alpha: float = 0.5) -> Path:
    """Persist the Grad-CAM overlay PNG (FR-17)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay = _overlay(saliency, np.ascontiguousarray(face_rgb), alpha=alpha)
    cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out_png