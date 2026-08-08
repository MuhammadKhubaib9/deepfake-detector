"""Performance metrics & charts for the developer dashboard.

Implements FR-18 (Accuracy, Precision, Recall, F1, ROC-AUC) and FR-19
(confusion matrix heatmap + ROC curve PNG charts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.linalg as _nl  # noqa: F401 (kept for sklearn metrics imports below)

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(y_true: np.ndarray, y_scores: np.ndarray,
             pos_label: int = 1) -> dict:
    """Compute the FR-18 metric set for binary classification."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "true_labels": [int(v) for v in y_true.tolist()],
        "predicted_scores": [round(float(v), 6) for v in y_scores.tolist()],
        "predicted_labels": [int(v) for v in y_pred.tolist()],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, digits=4,
            zero_division=0, output_dict=True,
        ),
    }


def roc_points(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr]}


def plot_confusion_matrix(cm: np.ndarray, labels, title: str,
                          out_png: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, model_name: str,
                   out_png: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_metrics_json(metrics: dict, out_json: str | Path) -> Path:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def load_predictions_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "true_labels": np.asarray(data["true_labels"]),
        "predicted_probs": np.asarray(data["predicted_probs"], dtype=float),
        "class_names": data.get("class_names", []),
    }