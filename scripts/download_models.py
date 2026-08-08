"""Download released model weights from Hugging Face (user's own repository).

Source:  https://huggingface.co/Khubaib7/deepfake-models
         (flat files: xception_weights.pt, vit_ffpp_weights.pth, cnn_lstm_weights.pth)

The three checkpoints (XceptionNet, ViT-B/16 FF++, CNN+BiLSTM) live in this
account's flat repo. The evaluation JSONs for the metrics dashboard are
produced by `scripts/generate_metrics.py` and are not downloaded here.

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
    "xception_weights.pt":  "models/xception_weights.pt",
    "vit_ffpp_weights.pth": "models/vit_ffpp_weights.pth",
    "cnn_lstm_weights.pth": "models/cnn_lstm_weights.pth",
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
    return 0


if __name__ == "__main__":
    sys.exit(main())