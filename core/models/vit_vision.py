"""ViT backbone - Vision Transformer B/16 (FR-12).

Loaded with the Hugging Face transformers library. The model files live in a
local folder (``models.vit.checkpoint``, e.g. ``./models/ViT``) so the
project is fully self-hosted and never touches the HF cache. If the local
folder is missing, it falls back to the owner's Hugging Face account
(``models.vit.hf_repo`` + ``hf_subfolder``) and downloads on first run.

Exposes ``predict_proba(frames_tensor) -> P(Fake)``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, resolve


class HuggingFaceViT(nn.Module):
    """ViTForImageClassification wrapper (transformers backend)."""

    def __init__(self, source, labels: tuple[str, str], **load_kwargs):
        super().__init__()
        from transformers import AutoModelForImageClassification

        self.model = AutoModelForImageClassification.from_pretrained(
            source, **load_kwargs
        )
        self.labels = list(labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224] ImageNet-normalised -> raw logits [B, 2]
        return self.model(pixel_values=x).logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]  # P(Fake)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        # Last transformer block (used for ViT explainability if requested)
        return self.model.vit.encoder.layer[-1]


def load_vit(cfg: Config, device=None) -> nn.Module:
    """Build the ViT model from config.yaml (local folder -> HF fallback)."""
    from . import get_device

    mcfg = cfg.models.vit
    local: Path | None = None
    checkpoint = mcfg.get("checkpoint")
    if checkpoint:
        local = resolve(checkpoint)
        if not (local / "config.json").is_file():
            local = None  # folder not present; fall back to the HF repo id
    if local is not None:
        model = HuggingFaceViT(str(local), labels=tuple(mcfg.labels))
    else:
        model = HuggingFaceViT(
            mcfg.hf_repo,
            labels=tuple(mcfg.labels),
            subfolder=mcfg.get("hf_subfolder"),
        )
    device = device or get_device(getattr(cfg, "models").device)
    model.to(device)
    model.eval()
    return model
