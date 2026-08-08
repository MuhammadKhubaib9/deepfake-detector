"""Generate the developer metrics dashboard artifacts (FR-18/FR-19).

Reads the released FF++ C23 test-set predictions (per model) downloaded by
``download_models.py``, computes Accuracy/Precision/Recall/F1/ROC-AUC per
model plus the weighted ensemble row, and writes:
  - metrics/metrics.json            (consumed by /api/metrics)
  - metrics/confusion_<model>.png
  - metrics/roc_<model>.png

Run:  python scripts/generate_metrics.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from core import metrics as mt  # noqa: E402
from core.config import load_config  # noqa: E402

MODELS = [
    ("cnn", "XceptionNet", "metrics/xception_test_predictions.json",
     "FaceForensics++ (C23)"),
    ("vit", "ViT-B/16 (FF++)", "metrics/vit_test_predictions.json",
     "FaceForensics++ (C23)"),
    ("lstm", "ResNet18-BiLSTM", "metrics/cnn_lstm_test_predictions.json",
     "FaceForensics++ (C23)"),
]

SRS_TARGETS = {  # NFR-03/04/05
    "accuracy": 0.75,
    "roc_auc": 0.85,
    "f1": 0.80,
}


def main():
    cfg = load_config()
    out_dir = Path(cfg.metrics.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble weights come from config.yaml (FR-14), the same video-path
    # weights used at inference time -> dashboard matches live behaviour.
    ens_weights = {
        k: float(v) for k, v in cfg.ensemble.video_weights.items()
    }
    print(f"ensemble weights from config: {ens_weights}")

    rows = {}
    arrays = {}

    for key, name, pred_file, dataset in MODELS:
        p = Path(pred_file)
        if not p.is_file():
            print(f"[skip] {name}: missing {p}")
            continue
        data = mt.load_predictions_json(p)
        y_true, y_scores = data["true_labels"], data["predicted_probs"]
        arrays[key] = (y_true, y_scores)

        ev = mt.evaluate(y_true, y_scores)
        cm = np.array(ev["confusion_matrix"])
        mt.plot_confusion_matrix(cm, ["Real", "Fake"], f"{name} - Confusion Matrix",
                                 out_dir / f"confusion_{key}.png")
        mt.plot_roc_curve(y_true, y_scores, name, out_dir / f"roc_{key}.png")

        rows[key] = {
            "model_name": name,
            "dataset": dataset,
            "metrics": {k: ev[k] for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "confusion_matrix": cm.tolist(),
            "chart_confusion": f"confusion_{key}.png",
            "chart_roc": f"roc_{key}.png",
            "targets_met": _check_targets(ev),
        }

    # ---- weighted ensemble row (soft voting, same 24-frame clip protocol)
    if all(k in arrays for k in ens_weights):
        (y_true0, _) = arrays["cnn"]
        n = len(y_true0)
        ok = all(len(arrays[k][0]) == n for k in ens_weights)
        if ok:
            total_w = sum(ens_weights.values())
            ens_scores = (sum(arrays[k][1] * w for k, w in ens_weights.items())
                          / total_w)
            ev = mt.evaluate(y_true0, ens_scores)
            cm = np.array(ev["confusion_matrix"])
            mt.plot_confusion_matrix(cm, ["Real", "Fake"],
                                     "Ensemble (CNN+ViT+LSTM) - Confusion Matrix",
                                     out_dir / "confusion_ensemble.png")
            mt.plot_roc_curve(y_true0, ens_scores, "Ensemble",
                              out_dir / "roc_ensemble.png")
            rows["ensemble"] = {
                "model_name": "Ensemble (CNN+ViT+LSTM)",
                "dataset": "FaceForensics++ (C23)",
                "metrics": {k: ev[k] for k in
                            ("accuracy", "precision", "recall", "f1", "roc_auc")},
                "confusion_matrix": cm.tolist(),
                "chart_confusion": "confusion_ensemble.png",
                "chart_roc": "roc_ensemble.png",
                "targets_met": _check_targets(ev),
            }

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "srs_targets": SRS_TARGETS,
        "note": ("Metrics computed on the released FF++ C23 test split "
                 "(per-model checkpoints from Hugging Face)."),
        "models": rows,
    }
    mt.save_metrics_json(doc, out_dir / "metrics.json")
    print(f"wrote {out_dir / 'metrics.json'} + charts for {list(rows)}")


def _check_targets(ev: dict) -> dict:
    return {
        "accuracy": ev["accuracy"] >= SRS_TARGETS["accuracy"],
        "roc_auc": ev["roc_auc"] >= SRS_TARGETS["roc_auc"],
        "f1": ev["f1"] >= SRS_TARGETS["f1"],
    }


if __name__ == "__main__":
    main()