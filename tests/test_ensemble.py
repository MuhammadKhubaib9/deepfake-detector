"""Ensemble weighted-soft-voting tests (FR-14..FR-16).

Active lineups (config ensemble.image_weights / video_weights):
  images:  cnn (XceptionNet), efficientnet, vit
  videos:  efficientnet, vit, vit_l14
Retired models (dima806 ViT-B/16, ResNet18-BiLSTM) have weight 0.0 and never
vote.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.ensemble import Ensemble  # noqa: E402

CFG = load_config()
ENS = Ensemble(CFG)


def run(scores, kind):
    return ENS.combine(dict(scores), kind)


def test_image_votes_use_only_cnn_effnet_vit():
    base = run({"cnn": 0.1, "efficientnet": 0.1, "vit": 0.1}, "image")
    assert base["verdict"] == "REAL"
    # Retired/other models must not move the image vote at all.
    padded = run({"cnn": 0.1, "efficientnet": 0.1, "vit": 0.1,
                  "vit_l14": 0.99, "lstm": 0.99, "vit_b16": 0.99}, "image")
    assert abs(base["p_fake"] - padded["p_fake"]) < 1e-9


def test_video_votes_use_only_effnet_vit_vitl14():
    base = run({"efficientnet": 0.9, "vit": 0.9, "vit_l14": 0.9}, "video")
    assert base["verdict"] == "FAKE"
    padded = run({"efficientnet": 0.9, "vit": 0.9, "vit_l14": 0.9,
                  "cnn": 0.01, "lstm": 0.99, "vit_b16": 0.99}, "video")
    assert abs(base["p_fake"] - padded["p_fake"]) < 1e-9


def test_single_strong_fake_vote_is_diluted():
    # Image lineup: one 0.99 vote next to two 0.05s -> weighted mean '0.36' -> REAL.
    r = run({"cnn": 0.05, "efficientnet": 0.05, "vit": 0.99}, "image")
    assert abs(r["p_fake"] - 0.3633) < 1e-3


def test_straddle_stays_real():
    # A single marginal 0.501 vote next to ~0.05 votes -> REAL, no INCONCLUSIVE.
    r = run({"cnn": 0.501, "efficientnet": 0.2036, "vit": 0.0531}, "image")
    assert r["verdict"] == "REAL"
    assert r["disagreement"] is False


def test_agreement_fake_image():
    r = run({"cnn": 0.9, "efficientnet": 0.87, "vit": 0.92}, "image")
    assert r["verdict"] == "FAKE"
    assert r["disagreement"] is False


def test_agreement_fake_video():
    r = run({"efficientnet": 0.8, "vit": 0.7, "vit_l14": 0.95}, "video")
    assert r["verdict"] == "FAKE"
    assert r["disagreement"] is False


def test_all_video_weights_are_equal():
    # Video lineup has three weight-1.0 members -> p_fake is a plain mean.
    r = run({"efficientnet": 0.2, "vit": 0.7, "vit_l14": 0.0}, "video")
    assert abs(r["p_fake"] - 0.3) < 1e-9
    assert r["verdict"] == "REAL"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}  -> {exc}")
    print(f"\nensemble: {passed}/{len(fns)} checks passed")
    sys.exit(0 if passed == len(fns) else 1)