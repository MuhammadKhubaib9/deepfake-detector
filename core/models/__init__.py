"""Model wrappers - CNN (XceptionNet), ViT (B/16), Temporal (ResNet18-BiLSTM).

Weight provenance (FF++ C23 checkpoints, hosted on the user's Hugging Face):
  - Khubaib7/deepfake-models  (xception_weights.pt, vit_ffpp_weights.pth, cnn_lstm_weights.pth)
"""
from __future__ import annotations

import torch

from . import cnn_xception, lstm_temporal, vit_vision


def get_device(value: str = "auto") -> torch.device:
    """Resolve the inference device (NFR-06: runs on CPU too)."""
    if value == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg, name: str):
    """Load a single model by short name: 'cnn' | 'vit' | 'lstm'."""
    if name == "cnn":
        return cnn_xception.load_cnn(cfg)
    if name == "vit":
        return vit_vision.load_vit(cfg)
    if name == "lstm":
        return lstm_temporal.load_temporal(cfg)
    raise ValueError(f"unknown model '{name}'")


class ModelBundle:
    """Container holding the loaded CNN / ViT / LSTM models + device."""

    def __init__(self, cnn=None, vit=None, lstm=None, device=None):
        self.cnn = cnn
        self.vit = vit
        self.lstm = lstm
        self.device = device

    def eval_all(self):
        for m in (self.cnn, self.vit, self.lstm):
            if m is not None:
                m.eval()

    def __repr__(self):
        parts = []
        if self.cnn is not None:
            parts.append(f"cnn={type(self.cnn).__name__}")
        if self.vit is not None:
            parts.append(f"vit={type(self.vit).__name__}")
        if self.lstm is not None:
            parts.append(f"lstm={type(self.lstm).__name__}")
        parts.append(f"device={self.device}")
        return f"ModelBundle({', '.join(parts)})"
