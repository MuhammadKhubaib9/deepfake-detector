"""Detection pipeline orchestrator (UC-03 image, UC-04 video).

Coordinates preprocessing -> CNN/ViT/LSTM inference -> ensemble -> Grad-CAM
and returns a SRS-compliant result document (DETECTION_RESULT).
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Config
from .ensemble import Ensemble
from .gradcam import compute_gradcam_heatmap, save_heatmap_overlay
from .models import ModelBundle, get_device
from .preprocessing import FacePreprocessor, NoFaceError, extract_video_frames


class Detector:
    """End-to-end deepfake inference for images and videos."""

    def __init__(self, cfg: Config, bundle: ModelBundle | None = None,
                 lock: threading.Lock | None = None,
                 device_override: str | None = None):
        self.cfg = cfg
        self.device = get_device(device_override or cfg.models.device)
        self.bundle = bundle or self._load_bundle(cfg, device_override)
        self.bundle.eval_all()
        self.pre = FacePreprocessor(cfg)
        self.ensemble = Ensemble(cfg)
        self.lock = lock or threading.RLock()

        prep = cfg.preprocessing
        self.fps = float(prep.video_fps)
        self.max_frames = int(prep.max_frames)
        self.max_duration = float(cfg.uploads.max_video_duration_seconds)

    # ------------------------------------------------------------ lifecycle
    @staticmethod
    def _load_bundle(cfg: Config, device_override: str | None = None) -> ModelBundle:
        from concurrent.futures import ThreadPoolExecutor
        from .models import cnn_efficientnet, cnn_xception, lstm_temporal, vit_vision

        device = get_device(device_override or cfg.models.device)
        jobs = {
            "cnn": lambda: cnn_xception.load_cnn(cfg, device=device),
            "effnet": lambda: cnn_efficientnet.load_effnet(cfg, device=device),
            "vit": lambda: vit_vision.load_vit(cfg, device=device),
            "lstm": lambda: lstm_temporal.load_temporal(cfg, device=device),
        }
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = {name: pool.submit(fn) for name, fn in jobs.items()}
            loaded = {name: fut.result() for name, fut in results.items()}
        return ModelBundle(cnn=loaded["cnn"], effnet=loaded["effnet"],
                           vit=loaded["vit"], lstm=loaded["lstm"],
                           device=device)

    # ------------------------------------------------------------- helpers
    def _to_clip_tensor(self, faces: list[np.ndarray]) -> torch.Tensor:
        """[T, 3, 224, 224] tensor from a list of aligned face crops."""
        batch = np.stack([self.pre.face_to_tensor(f) for f in faces], axis=0)
        return torch.from_numpy(batch).to(self.device)

    @staticmethod
    def _persist_png(rgb: np.ndarray, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return path

    @staticmethod
    def _artifacts_dir(artifacts_dir) -> Path:
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --------------------------------------------------------------- image
    def detect_image(self, image_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-03: CNN + ViT inference on one image, with Grad-CAM."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise RuntimeError("Could not read the uploaded image.")

            face, box, conf = self.pre.detect_face(bgr)          # FR-06..FR-08
            clip = self._to_clip_tensor([face])             # [1,3,224,224]

            with torch.no_grad(), torch.inference_mode():
                p_cnn = torch.sigmoid(self.bundle.cnn(clip)).item()
                p_effnet = torch.sigmoid(self.bundle.effnet(clip)).item()
                p_vit = float(self.bundle.vit.predict_proba(clip).item())

            saliency = compute_gradcam_heatmap(self.bundle.cnn, clip, self.device)
            heat_p = save_heatmap_overlay(saliency, face, artifacts / "heatmap.png")
            crop_p = self._persist_png(face, artifacts / "face_crop.png")

            result = self.ensemble.combine(
                {"cnn": p_cnn, "efficientnet": p_effnet, "vit": p_vit},
                kind="image")
            return self._finalize(result, kind="image", artifacts=artifacts,
                                  original_name=original_name,
                                  heatmap_path=heat_p.name,
                                  face_crop_path=crop_p.name,
                                  frames_analyzed=1)

    # --------------------------------------------------------------- video
    def detect_video(self, video_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-04: frames -> CNN/ViT frame scores + LSTM temporal score."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            frames = extract_video_frames(
                video_path, fps=self.fps, max_seconds=self.max_duration,
                max_frames=self.max_frames, tmp_dir=artifacts / "frames",
            )
            if not frames:
                raise RuntimeError(
                    "Video processing failed. Please try a different file format."
                )

            faces: list[np.ndarray] = []
            for fp in frames:
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                try:
                    face, _, _ = self.pre.detect_face(bgr)
                    faces.append(face)
                except NoFaceError:
                    continue  # per-frame rejection logged (P2.2)

            if not faces:
                raise NoFaceError(
                    "No face detected. Please upload media containing a "
                    "clearly visible human face."
                )

            clip = self._to_clip_tensor(faces)                  # [T,3,224,224]
            with torch.no_grad(), torch.inference_mode():
                cnn_logits = self.bundle.cnn(clip)              # [T,1]
                p_cnn_frames = torch.sigmoid(cnn_logits).cpu().numpy().reshape(-1)
                p_cnn = float(p_cnn_frames.mean())
                eff_logits = self.bundle.effnet(clip)           # [T,1]
                p_eff_frames = torch.sigmoid(eff_logits).cpu().numpy().reshape(-1)
                p_effnet = float(p_eff_frames.mean())
                p_lstm = float(torch.sigmoid(self.bundle.lstm(clip)).item())
                p_vit = float(self.bundle.vit.predict_proba(clip).mean().item())

            # Grad-CAM on the most manipulated frame (FR-17).
            # No inference_mode here: Grad-CAM needs autograd.
            worst = int(np.argmax(p_cnn_frames))
            t_worst = self._to_clip_tensor([faces[worst]])  # [1,3,224,224]
            saliency = compute_gradcam_heatmap(self.bundle.cnn, t_worst, self.device)
            heat_p = save_heatmap_overlay(saliency, faces[worst].astype(np.uint8),
                                          artifacts / "heatmap.png")
            crop_p = self._persist_png(faces[0], artifacts / "face_crop.png")

            result = self.ensemble.combine(
                {"cnn": p_cnn, "efficientnet": p_effnet, "vit": p_vit,
                 "lstm": p_lstm},
                kind="video"
            )
            return self._finalize(
                result, kind="video", artifacts=artifacts,
                original_name=original_name, heatmap_path=heat_p.name,
                face_crop_path=crop_p.name, frames_analyzed=len(faces),
                most_manipulated_frame=worst + 1,
                analyzed_seconds=round(len(frames) / self.fps, 1),
            )

    # ---------------------------------------------------------------- util
    def _finalize(self, result: dict, *, kind: str, artifacts: Path,
                  original_name: str | None, heatmap_path: str,
                  face_crop_path: str, frames_analyzed: int, **extra) -> dict:
        doc = {
            "kind": kind,
            "verdict": result["verdict"],
            "p_fake": result["p_fake"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "scores": result["scores"],
            "cnn_score": result["scores"].get("cnn"),
            "effnet_score": result["scores"].get("efficientnet"),
            "vit_score": result["scores"].get("vit"),
            "lstm_score": result["scores"].get("lstm") if kind == "video" else None,
            "heatmap_path": heatmap_path,
            "face_crop_path": face_crop_path,
            "faces_analyzed": frames_analyzed,
            "video_fps": self.fps if kind == "video" else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc.update(extra)
        if original_name is not None:
            doc["original_name"] = original_name
        return doc
