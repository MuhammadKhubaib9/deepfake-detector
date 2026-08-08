"""Temporal module - ResNet18 + 2-layer BiLSTM (FR-13).

Loads the FF++ C23 CNN+BiLSTM checkpoint (Hugging Face:
Khubaib7/deepfake-models). Per-frame CNN features feed
the BiLSTM => P(Fake) for a clip of T frames.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, resolve


class TemporalNet(nn.Module):
    """CNN (ResNet-18, no avgpool/fc) -> BiLSTM(hidden=256, 2 layers) -> head."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        from torchvision.models import resnet18

        base = resnet18(weights=None)
        base.fc = nn.Identity()   # binary head lives in self.classifier
        self.cnn_feature_extractor = base  # outputs [B*T, 512]

        self.bilstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, 3, 224, 224] ImageNet-normalised clip -> scalar logit."""
        if x.dim() == 4:                     # [T, C, H, W]
            x = x.unsqueeze(0)               # [1, T, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"expected [T,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        cnn_in = x.reshape(B * T, C, H, W)
        feats = self.cnn_feature_extractor(cnn_in)  # [B*T, 512]
        feats = feats.view(B, T, -1)                # [B, T, 512]
        out, _ = self.bilstm(feats)              # [B, T, 512]
        final = out[:, -1, :]                    # last timestep
        return self.classifier(final)            # [B, 1]


def load_temporal(cfg: Config, device=None):
    """Load the released FF++ CNN+BiLSTM checkpoint."""
    from . import get_device

    mcfg = cfg.models.lstm
    ckpt_path = resolve(mcfg.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"LSTM checkpoint missing: {ckpt_path}\n"
            "Run: python scripts/download_models.py"
        )

    model = TemporalNet(hidden_size=int(mcfg.hidden_size),
                        num_layers=int(mcfg.num_layers),
                        dropout=float(mcfg.dropout))
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing LSTM weights: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected LSTM weights: {unexpected[:10]}")

    device = device or get_device(getattr(cfg, "models").device)
    model.to(device)
    model.eval()
    return model


def clip_logit_to_proba(logit: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logit.squeeze(-1))