"""EfficientNet-B3 backbone (spatial CNN, frame-level).

Loads the FF++ C23 fine-tuned EfficientNet-B3 checkpoint (Hugging Face:
MUmairAB/deepfake-detection-ff-cn-transformer, efficientnet_b3_baseline),
mirroring the released timm architecture. Outputs a single sigmoid logit =>
P(Fake).

This is the strongest single model in the FF++ C23 study (AUC 0.9976,
accuracy 0.9829, best held-out-FaceShifter generalization), so it is used as
the second spatial backbone alongside XceptionNet.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, resolve

_PREFIX = "backbone."


class EfficientNetNet(nn.Module):
    """Wrapper exposing the timm EfficientNet-B3 backbone under ``.backbone``.

    ``.backbone`` is kept as a named submodule so the released checkpoints
    (whose keys are prefixed ``backbone.``) load verbatim.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        import timm

        self.backbone = timm.create_model("efficientnet_b3",
                                          pretrained=False,
                                          num_classes=num_classes)
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """Final convolutional block (used for Grad-CAM if requested)."""
        return self.backbone.blocks[-1][-1]


def load_effnet(cfg: Config, device=None):
    """Instantiate EfficientNetNet and load the released FF++ checkpoint."""
    from . import get_device

    mcfg = cfg.models.efficientnet
    ckpt_path = resolve(mcfg.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"EfficientNet checkpoint missing: {ckpt_path}\n"
            "Run: python scripts/download_models.py"
        )

    model = EfficientNetNet(num_classes=1)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw

    # ckpt keys are prefixed with the wrapper's `backbone.` submodule name,
    # which matches our EfficientNetNet layout -> load verbatim.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing EfficientNet weights: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected EfficientNet weights: {unexpected[:10]}")

    device = device or get_device(getattr(cfg, "models").device)
    model.to(device)
    model.eval()
    return model