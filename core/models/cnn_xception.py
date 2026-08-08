"""CNN backbone - XceptionNet (FR-11).

Loads the FF++ C23 fine-tuned Xception checkpoint (Hugging Face:
Khubaib7/deepfake-models) on top of timm's legacy
Xception. Outputs a single sigmoid logit => P(Fake).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, resolve

_PREFIX = "backbone."


class XceptionNet(nn.Module):
    """Wrapper exposing the timm Xception backbone under ``.backbone``.

    ``.backbone`` is kept as a named submodule so pytorch-grad-cam can hook
    the final convolutional block (``backbone.conv4``).
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        import timm

        if hasattr(timm, "create_model"):
            try:  # timm >= 1.0
                self.backbone = timm.create_model("legacy_xception",
                                                  pretrained=False,
                                                  num_classes=num_classes)
            except Exception:  # timm 0.9.x
                self.backbone = timm.create_model("xception",
                                                  pretrained=False,
                                                  num_classes=num_classes)
        else:
            raise RuntimeError("timm is required for the Xception CNN")

        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """Final convolutional block used for Grad-CAM (FR-17)."""
        return self.backbone.conv4


def load_cnn(cfg: Config, device=None):
    """Instantiate XceptionNet and load the released FF++ checkpoint."""
    from . import get_device

    mcfg = cfg.models.cnn
    ckpt_path = resolve(mcfg.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"CNN checkpoint missing: {ckpt_path}\n"
            "Run: python scripts/download_models.py"
        )

    model = XceptionNet(num_classes=1)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    else:
        sd = raw

    # ckpt keys are prefixed with the wrapper's `backbone.` submodule name,
    # which matches our XceptionNet layout -> load verbatim.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing Xception weights: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected Xception weights: {unexpected[:10]}")

    device = device or get_device(getattr(cfg, "models").device)
    model.to(device)
    model.eval()
    return model
