"""CommunityForensics ViT-Small - cross-generator fake image detector.

Trained on 2.7M samples across 4,803 generators (CommunityForensics,
Park & Owens, CVPR 2025). Standard transformers image-classification head
(num_labels=1, sigmoid logit => P(fake)). A local clone of
`buildborderless/CommunityForensics-DeepfakeDet-ViT` is used when present;
otherwise the owner's Hugging Face repo mirrors it (local-first, NFR-07).
Requires `transformers >= 5.4` for the
shortest-edge-440 / center-crop-384 / CLIP-norm preprocessor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_TARGET = 384
_SHORTEST = 440


class CommunityVitNet(nn.Module):
    """CommunityForensics ViT over 384x384 CLIP-normalised face crops.

    Inputs are uint8 RGB face crops (any size); the wrapper re-sizes the
    shortest edge to 440 and centre-crops 384 before the CLIP-style norm
    (>required by the model card).
    """

    def __init__(self, source, **load_kwargs):
        super().__init__()
        from transformers import ViTForImageClassification

        self.backbone = ViTForImageClassification.from_pretrained(source,
                                                                  **load_kwargs)
        self.target_size = _TARGET

    @torch.no_grad()
    def predict_proba(self, faces_rgb: list[np.ndarray]) -> np.ndarray:
        """[N] P(fake) sigmoid probabilities from uint8 face crops."""
        import cv2

        batch = []
        for face in faces_rgb:
            h, w = face.shape[:2]
            scale = _SHORTEST / min(h, w)
            resized = cv2.resize(face, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            top = max(0, (rh - _TARGET) // 2)
            left = max(0, (rw - _TARGET) // 2)
            crop = resized[top:top + _TARGET, left:left + _TARGET]
            t = (crop.astype(np.float32) / 255.0 - _MEAN) / _STD
            batch.append(np.transpose(t, (2, 0, 1)))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(self.device)
        logits = self.backbone(x).logits
        return torch.sigmoid(logits.squeeze(-1)).cpu().numpy().reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).logits


def load_community_vit(cfg: Config, device=None) -> CommunityVitNet:
    """Instantiate the CommunityForensics ViT on `device`.

    Local copy (``models.community.checkpoint``) is used when present;
    otherwise the owner's Hugging Face repo is used as a download fallback
    (``models.community.hf_repo`` + ``hf_subfolder``) - same pattern as ViT.
    """
    from . import get_device

    mcfg = cfg.models.community
    local = resolve(mcfg.get("checkpoint"))
    if (local / "config.json").is_file():
        model = CommunityVitNet(str(local))
    else:
        model = CommunityVitNet(mcfg.hf_repo,
                                subfolder=mcfg.get("hf_subfolder"))
    model.device = device or get_device(getattr(cfg, "models").device)
    model.to(model.device)
    model.eval()
    return model