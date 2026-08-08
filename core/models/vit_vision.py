"""ViT backbone - Vision Transformer B/16 (FR-12).

Two backends are supported:
  * ``ffpp`` (default): local FF++ C23 fine-tuned ViT-B/16 checkpoint (timm),
    downloaded from Khubaib7/deepfake-models.
  * ``transformers``: Khubaib7/deepfake-models via the Hugging Face
    transformers library (AutoModelForImageClassification). Note: a proper
    transformers model repo (config.json + weights) is required; set
    ``backend: transformers`` in config.yaml to use it.

Both expose ``predict_proba(frames_tensor) -> P(Fake)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve

_FFPP_PREFIX = "backbone."


class HuggingFaceViT(nn.Module):
    """ViTForImageClassification wrapper (transformers backend)."""

    def __init__(self, model_id: str, labels: tuple[str, str]):
        super().__init__()
        from transformers import AutoModelForImageClassification

        self.model = AutoModelForImageClassification.from_pretrained(model_id)
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


class FfppViT(nn.Module):
    """Local FF++ ViT-B/16 checkpoint (timm) - P(Fake) via sigmoid."""

    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model("vit_base_patch16_224",
                                          pretrained=False,
                                          num_classes=1)
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x).squeeze(-1))


def load_vit(cfg: Config, device=None) -> nn.Module:
    """Build the ViT model chosen in config.yaml."""
    from . import get_device

    mcfg = cfg.models.vit
    backend = getattr(mcfg, "backend", "transformers")

    if backend == "ffpp":
        ckpt_path = resolve(mcfg.ffpp_checkpoint)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"ViT (ffpp) checkpoint missing: {ckpt_path}\n"
                "Run: python scripts/download_models.py"
            )
        model = FfppViT()
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
        # ckpt keys are prefixed with the wrapper's `backbone.` submodule name,
        # matching our FfppViT layout -> load verbatim.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"ViT (ffpp) state dict mismatch: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
    else:
        model = HuggingFaceViT(
            model_id=mcfg.hf_weights,
            labels=tuple(mcfg.labels),
        )

    device = device or get_device(getattr(cfg, "models").device)
    model.to(device)
    model.eval()
    return model
