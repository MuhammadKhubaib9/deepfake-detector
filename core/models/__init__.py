"""Model wrappers - spatial CNNs (XceptionNet, EfficientNet-B3), ViT (B/16),
and Temporal (ResNet18-BiLSTM).

Weight provenance (FF++ C23 checkpoints, hosted on the user's Hugging Face):
  - Khubaib7/deepfake-models  (xception_weights.pt, cnn_lstm_weights.pth, efficientnet_weights.pt)
  - dima806/deepfake_vs_real_image_detection  (ViT-B/16, transformers-native)
"""
from __future__ import annotations

import torch

from . import cnn_efficientnet, cnn_xception, lstm_temporal, vit_lnclip, vit_community, vit_vision


def get_device(value: str = "auto") -> torch.device:
    """Resolve the inference device (NFR-06: runs on CPU too)."""
    if value == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg, name: str):
    """Load a single model by short name: 'cnn' | 'effnet' | 'vit' | 'vit_l14'."""
    if name == "cnn":
        return cnn_xception.load_cnn(cfg)
    if name == "effnet":
        return cnn_efficientnet.load_effnet(cfg)
    if name == "vit":
        return vit_community.load_community_vit(cfg)
    if name == "vit_l14":
        return vit_lnclip.load_lnclip(cfg)
    raise ValueError(f"unknown model '{name}'")


class ModelBundle:
    """Container holding the loaded Xception / EfficientNet / ViT / ViT-L/14
    models + device. Retired wrappers (dima806 ViT-B/16, BiLSTM) are simply
    never constructed."""

    def __init__(self, cnn=None, effnet=None, vit=None, vit_l14=None, device=None):
        self.cnn = cnn
        self.effnet = effnet
        self.vit = vit
        self.vit_l14 = vit_l14
        self.device = device

    def eval_all(self):
        for m in (self.cnn, self.effnet, self.vit, self.vit_l14):
            if m is not None:
                m.eval()

    def __repr__(self):
        parts = []
        if self.cnn is not None:
            parts.append(f"cnn={type(self.cnn).__name__}")
        if self.effnet is not None:
            parts.append(f"effnet={type(self.effnet).__name__}")
        if self.vit is not None:
            parts.append(f"vit={type(self.vit).__name__}")
        if self.vit_l14 is not None:
            parts.append(f"vit_l14={type(self.vit_l14).__name__}")
        parts.append(f"device={self.device}")
        return f"ModelBundle({', '.join(parts)})"
