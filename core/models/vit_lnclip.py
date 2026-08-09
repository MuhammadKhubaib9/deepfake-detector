"""LNCLIP-DF - generalizable deepfake CLIP detector (WACV 2026).

Weights: `yermandy/deepfake-detection` model.torchscript (ViT-L/14 visual
encoder, LN-tuned on FF++). Top generalisation on unseen generators
(DFDC 87.2 / Celeb-DF-v2 96.6 video AUROC in the paper's survey).

Inference mirror the author's `inference_torchscript.py`:
  - CLIP ViT-L/14 preprocessing: resize shortest edge to 224, centre-crop
    224, CLIP mean/std normalisation (applied offline, no Hub processor).
  - The traced model returns 2-class logits [B,2] (real, fake) -> softmax.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_TARGET = 224


class LnClipNet(nn.Module):
    """TorchScript-wrapped CLIP ViT-L/14 with an LN-tuned head."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.target_size = _TARGET
        self.backbone = torch.jit.load(str(resolve(ckpt_path)),
                                       map_location="cpu")

    @torch.no_grad()
    def predict_proba(self, faces_rgb: list[np.ndarray]) -> np.ndarray:
        """[P] P(fake) probabilities from uint8 face crops (CLIP-norm 224)."""
        batch = []
        for face in faces_rgb:
            h, w = face.shape[:2]
            scale = _TARGET / min(h, w)
            resized = cv2.resize(face, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            top = max(0, (rh - _TARGET) // 2)
            left = max(0, (rw - _TARGET) // 2)
            crop = resized[top:top + _TARGET, left:left + _TARGET]
            t = (crop.astype(np.float32) / 255.0 - _MEAN) / _STD
            batch.append(np.transpose(t, (2, 0, 1)))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(self.device)
        logits = self.backbone(x)  # [B, 2] (real, fake)
        probs = torch.softmax(logits.float(), dim=1)
        return probs[:, 1].cpu().numpy().reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_lnclip(cfg: Config, device=None) -> LnClipNet:
    """Load the traced ViT-L/14 (LNCLIP) model on `device`.

    Local copy (``models.vit_l14.checkpoint``) is used when present;
    otherwise the file is pulled from the owner's Hugging Face repo
    (``models.vit_l14.hf_repo`` + ``hf_subfolder``).
    """
    from . import get_device

    mcfg = cfg.models.vit_l14
    local = resolve(mcfg.get("checkpoint"))
    ckpt_path = local
    if not local.is_file():
        from huggingface_hub import hf_hub_download
        ckpt_path = Path(hf_hub_download(
            repo_id=mcfg.hf_repo,
            filename=mcfg.get("file"),
            subfolder=mcfg.get("hf_subfolder"),
        ))
    model = LnClipNet(str(ckpt_path))
    model.device = get_device(getattr(cfg, "models").device)
    model.to(model.device)
    model.eval()
    return model