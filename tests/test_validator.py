"""SRS traceability tests - upload validation rules (FR-01..FR-04)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.validator import FileValidator, ValidationError  # noqa: E402

CFG = load_config()
V = FileValidator(CFG)

# minimal valid magic-byte prefixes
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 56
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 53
AVI = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 50
GIF = b"GIF89a" + b"\x00" * 57


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError:
        print(f"FAIL  {name}")


def t_accept_jpeg():
    m = V.validate("photo.jpg", JPEG)
    assert m["kind"] == "image" and m["extension"] == "jpeg"


def t_accept_png():
    m = V.validate("photo.png", PNG)
    assert m["kind"] == "image" and m["extension"] == "png"


def t_accept_mp4():
    m = V.validate("clip.mp4", MP4)
    assert m["kind"] == "video" and m["extension"] == "mp4"


def t_accept_avi():
    m = V.validate("clip.avi", AVI)
    assert m["kind"] == "video" and m["extension"] == "avi"


def t_reject_gif():
    try:
        V.validate("a.gif", GIF)
    except ValidationError as e:
        assert "Unsupported format" in str(e)
        return
    raise AssertionError("gif should be rejected")


def t_reject_mismatched_ext():
    try:
        V.validate("fake.jpg", PNG)  # PNG bytes, .jpg name
    except ValidationError as e:
        assert "do not match" in str(e)
        return
    raise AssertionError("mismatch should be rejected")


def t_reject_large_image():
    big = JPEG + b"\x00" * (11 * 1024 * 1024)
    try:
        V.validate("big.jpg", big)
    except ValidationError as e:
        assert "too large" in str(e) and "10 MB" in str(e)
        return
    raise AssertionError("11 MB image should be rejected")


def t_reject_large_video():
    big = MP4 + b"\x00" * (101 * 1024 * 1024)
    try:
        V.validate("big.mp4", big)
    except ValidationError as e:
        assert "too large" in str(e) and "100 MB" in str(e)
        return
    raise AssertionError("101 MB video should be rejected")


def t_reject_empty():
    try:
        V.validate("empty.jpg", b"")
    except ValidationError:
        return
    raise AssertionError("empty file should be rejected")


def t_exact_image_limit_ok():
    m = V.validate("limit.jpg", JPEG + b"\x00" * (10 * 1024 * 1024 - len(JPEG)))
    assert m["size"] <= V.max_image_bytes


if __name__ == "__main__":
    check("FR-01 JPEG accepted", t_accept_jpeg)
    check("FR-01 PNG accepted", t_accept_png)
    check("FR-02 MP4 accepted", t_accept_mp4)
    check("FR-02 AVI accepted", t_accept_avi)
    check("FR-03 GIF rejected", t_reject_gif)
    check("FR-03 magic/extension mismatch rejected", t_reject_mismatched_ext)
    check("FR-04 11 MB image rejected", t_reject_large_image)
    check("FR-04 101 MB video rejected", t_reject_large_video)
    check("empty file rejected", t_reject_empty)
    check("FR-04 10 MB image boundary ok", t_exact_image_limit_ok)