"""File upload validation (P1.0 Media Input Handler / File Validator).

Implements FR-01..FR-04:
  FR-01  image uploads JPEG/PNG only
  FR-02  video uploads MP4/AVI only, max 100 MB
  FR-03  validate MIME type AND file header (magic bytes); reject mismatches
  FR-04  size limits (images 10 MB, videos 100 MB)

Error messages mirror the SRS strings (Section 3.1).
"""
from __future__ import annotations

import mimetypes as _mimetypes

from .config import Config

_HEADER_BYTES = 64

_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_AVI = b"RIFF"
_MP4_BRANDS = (b"isom", b"mp41", b"mp42", b"av01", b"iso6", b"MSNV", b"cm")

IMAGE_EXT = frozenset({"jpeg", "png"})
VIDEO_EXT = frozenset({"mp4", "avi"})

EXT_MIME = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
}

MIME_KIND = {
    "image/jpeg": "image",
    "image/png": "image",
    "video/mp4": "video",
    "video/x-msvideo": "video",
}


class ValidationError(Exception):
    """Raised when an uploaded file violates FR-01..FR-04."""


def sniff_extension(header: bytes) -> str | None:
    """Detect the real container/image format from magic bytes (FR-03)."""
    if not header:
        return None
    if header.startswith(_JPEG):
        return "jpeg"
    if header.startswith(_PNG):
        return "png"
    if header[4:8] in _MP4_BRANDS or header[8:12] in _MP4_BRANDS:
        # MP4 family: 'ftyp' box at offset 4 with a known brand at 8..12
        return "mp4"
    if header.startswith(_AVI) and header[8:12] == b"AVI ":
        return "avi"
    return None


def canonical_ext(ext: str) -> str:
    return {"jpeg": "jpeg", "jpg": "jpeg"}.get(ext.lower(), ext.lower())


def mime_for(ext: str) -> str | None:
    return EXT_MIME.get(ext.lower())


def kind_for_mime(mime: str) -> str | None:
    return MIME_KIND.get(mime)


class FileValidator:
    """Validates an uploaded file against the SRS upload palette."""

    def __init__(self, cfg: Config):
        up = cfg.uploads
        self.max_image_bytes = int(up.max_image_size_mb) * 1024 * 1024
        self.max_video_bytes = int(up.max_video_size_mb) * 1024 * 1024
        self.allowed_image = frozenset(up.allowed_image_types)
        self.allowed_video = frozenset(up.allowed_video_types)

    def validate(self, filename: str, data: bytes) -> dict:
        """Validate bytes + name; return SRS-compliant metadata or raise.

        Returns a dict with keys: kind, extension, mime, size, original_name.
        """
        if not data:
            raise ValidationError("Empty file. Please try again.")

        ext = sniff_extension(data[: _HEADER_BYTES])
        if ext is None:
            raise ValidationError("Unsupported format. Accepted: JPEG, PNG, MP4, AVI.")

        mime = EXT_MIME[ext]
        kind = MIME_KIND[mime]

        # FR-03: reject extension/content mismatches (e.g. renamed .exe, .jpg
        # that is really a PNG...). Compare guessed MIME against magic bytes.
        declared_mime, _ = _mimetypes.guess_type(filename)
        if declared_mime is not None and declared_mime in EXT_MIME.values():
            if declared_mime != mime:
                raise ValidationError("File contents do not match its extension.")

        allowed = self.allowed_image if kind == "image" else self.allowed_video
        if mime not in allowed:
            raise ValidationError("Unsupported format. Accepted: JPEG, PNG, MP4, AVI.")

        if kind == "image":
            if len(data) > self.max_image_bytes:
                raise ValidationError("File too large. Maximum: 10 MB for images.")
        elif len(data) > self.max_video_bytes:
            raise ValidationError("File too large. Maximum: 100 MB for videos.")

        return {
            "kind": kind,
            "extension": ext,
            "mime": mime,
            "size": len(data),
            "original_name": filename,
        }