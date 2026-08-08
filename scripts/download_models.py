"""Download model weights from the project's own Hugging Face repository.

Sources:
  * Khubaib7/deepfake-models  (flat files):
      - xception_weights.pt        FF++ C23 XceptionNet
      - cnn_lstm_weights.pth       FF++ C23 ResNet18-BiLSTM
      - efficientnet_weights.pt    FF++ C23 EfficientNet-B3
  * ViT-B/16 lives locally in models/ViT/ (self-hosted copy of
      dima806/deepfake_vs_real_image_detection) - nothing to download.

Run:  python scripts/download_models.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "Khubaib7/deepfake-models"

FILES = {
    "xception_weights.pt":      "models/xception_weights.pt",
    "cnn_lstm_weights.pth":     "models/cnn_lstm_weights.pth",
    "efficientnet_weights.pt":  "models/efficientnet_weights.pt",
}


def _download(src: str, dst_path: Path) -> None:
    """Download one repo file into a scratch dir, then move it into place."""
    tmp = Path(tempfile.mkdtemp(prefix="hf_dl_"))
    try:
        hf_hub_download(repo_id=REPO, filename=src, local_dir=str(tmp))
        target = tmp / src
        if not target.is_file():
            hits = list(tmp.rglob(src))
            target = hits[0] if hits else None
        if target is None or not target.is_file():
            raise RuntimeError(f"downloaded file not found: {src}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dst_path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    for src, dst in FILES.items():
        dst_path = ROOT / dst
        if dst_path.is_file() and dst_path.stat().st_size > 0:
            print(f"[present ] {dst}")
            continue
        print(f"[download] {REPO}/{src}")
        _download(src, dst_path)
        print(f"[done]    {dst}")
    print("ViT (dima806/deepfake_vs_real_image_detection) downloads "
          "automatically on first detection run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())