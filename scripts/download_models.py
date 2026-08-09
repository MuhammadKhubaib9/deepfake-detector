"""Download model weights from the project's own Hugging Face repository.

Source repo:  https://huggingface.co/Khubaib7/deepfake-models

Layout inside the repo (paths relative to the repo root, after the owner's
HF-side rename of the two active ViT models):

  xception_weights.pt                 FF++ C23 XceptionNet
  cnn_lstm_weights.pth                FF++ C23 ResNet18-BiLSTM (retired)
  efficientnet_weights.pt             FF++ C23 EfficientNet-B3
  ViT/                                ViT model (CommunityForensics ViT-Small, CVPR 2025)
  vit_l14/model.torchscript           ViT-L/14 (LNCLIP-DF traced CLIP ViT-L/14, WACV 2026)
  ViT-B16-retired/                    retired dima806 ViT-B/16 (optional local copy)

Run:  python scripts/download_models.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "Khubaib7/deepfake-models"

FILES = {
    "xception_weights.pt":      "models/xception_weights.pt",
    "cnn_lstm_weights.pth":     "models/cnn_lstm_weights.pth",
    "efficientnet_weights.pt":  "models/efficientnet_weights.pt",
}

# Folder checkpoints downloaded whole from a repo subfolder into models/<name>.
FOLDERS = {
    "ViT":                  "models/ViT",                  # active ViT (CVPR 2025)
    "vit_l14":              "models/vit_l14",              # ViT-L/14 torchscript
    "ViT-B16-retired":      "models/ViT-B16-retired",      # retired dima806 ViT-B/16
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


def _download_folder(folder: str, dst_path: Path) -> None:
    """Mirror a repo subfolder (e.g. ViT/) into the local models dir."""
    if (dst_path / "config.json").is_file() or (dst_path / "model.torchscript").is_file():
        print(f"[present ] {dst_path}")
        return
    tmp = Path(tempfile.mkdtemp(prefix="hf_dl_"))
    try:
        print(f"[download] {REPO}/{folder}  ->  {dst_path}")
        snapshot_download(repo_id=REPO, allow_patterns=[f"{folder}/*"],
                          local_dir=str(tmp))
        src_dir = tmp / folder
        if not src_dir.is_dir():
            raise RuntimeError(
                f"subfolder not found in repo: {folder}. Please re-upload the "
                f"renamed checkpoint to {REPO}/{folder} first."
            )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            shutil.rmtree(dst_path, ignore_errors=True)
        shutil.move(str(src_dir), str(dst_path))
        print(f"[done]    {dst_path}")
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
    for folder, dst in FOLDERS.items():
        _download_folder(folder, ROOT / dst)
    if (ROOT / "models" / "ViT" / "config.json").is_file():
        print("ViT ready.")
    if (ROOT / "models" / "vit_l14" / "model.torchscript").is_file():
        print("ViT-L/14 torchscript ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())