"""Ensemble weighted-soft-voting tests (FR-14..FR-16)."""
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


def test_unanimous_vote_stays_real():
    r = run({"cnn": 0.38, "efficientnet": 0.26, "vit": 0.38, "lstm": 0.02}, "video")
    assert r["verdict"] == "REAL"
    assert r["disagreement"] is False


def test_mixed_vote_averages_to_real():
    # Despite a strong ViT spike, the weighted average (ViT dropped from video)
    # is far below the threshold.
    r = run({"cnn": 0.06, "efficientnet": 0.04, "vit": 0.99, "lstm": 0.03}, "video")
    assert r["verdict"] == "REAL"
    assert r["disagreement"] is False


def test_vit_is_excluded_from_video_votes():
    # Video weights no longer include 'vit' -> a 0.99 ViT vote must NOT move
    # the weighted average (weight 0 drops it from both numerator & denominator).
    without_vit = run({"cnn": 0.06, "efficientnet": 0.04, "lstm": 0.03}, "video")
    with_vit = run({"cnn": 0.06, "efficientnet": 0.04, "lstm": 0.03, "vit": 0.99},
                   "video")
    assert abs(without_vit["p_fake"] - with_vit["p_fake"]) < 1e-9
    assert with_vit["verdict"] == "REAL"


def test_vit_keeps_voting_on_images():
    # Image weights still include the ViT -> its vote moves the image verdict.
    r = run({"cnn": 0.45, "efficientnet": 0.45, "vit": 0.99}, "image")
    assert r["verdict"] == "FAKE"
    r2 = run({"cnn": 0.45, "efficientnet": 0.45}, "image")
    assert r2["verdict"] == "REAL"


def test_straddle_stays_real():
    # A single marginal 0.501 vote next to 0.05 votes -> REAL (no INCONCLUSIVE).
    r = run({"cnn": 0.501, "efficientnet": 0.2036, "vit": 0.0531}, "image")
    assert r["verdict"] == "REAL"
    assert r["disagreement"] is False


def test_agreement_fake():
    r = run({"cnn": 0.9, "efficientnet": 0.87, "vit": 0.92}, "image")
    assert r["verdict"] == "FAKE"
    assert r["disagreement"] is False


def test_weights_are_respected():
    # Equal p but video weight on lstm is 0.7 -> shifts the average slightly.
    r = run({"cnn": 0.8, "efficientnet": 0.7, "vit": 0.7, "lstm": 0.1}, "video")
    assert 0.5 < r["p_fake"] < 0.8


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