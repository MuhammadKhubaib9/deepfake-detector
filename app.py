"""DeepFake Detection System - Flask backend (Tier 1 web layer).

Routes (SRS Sections 3 & 5):
  GET  /                        upload screen (P1) + results (P2)
  GET  /metrics                 developer metrics dashboard (UC-07, P3)
  POST /api/upload              validate + store a media file (UC-01/UC-02)
  POST /api/detect              run the AI pipeline, return verdict (UC-03/UC-04)
  GET  /media/<sid>/<file>      serve session-scoped artifacts
  GET  /metrics-chart/<name>    serve dashboard PNG charts (FR-19)
  GET  /api/metrics             FR-18 metrics JSON (dashboard data)
  GET  /api/status              readiness probe

Compliance: FR-01..FR-04 validation & SRS messages, FR-20 expiry sweep, NFR-07
fully local inference, Data-01/02 in-memory session store.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from core.config import load_config, resolve
from core.detector import Detector
from core.preprocessing import NoFaceError
from core.validator import FileValidator, ValidationError

CFG = load_config()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(CFG.uploads.max_video_size_mb) * 1024 * 1024
APP_SESSION_EXPIRY = int(CFG.uploads.session_expiry_seconds)

UPLOAD_ROOT = resolve(CFG.uploads.upload_dir)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_DIR = resolve(CFG.metrics.output_dir)

validator = FileValidator(CFG)

sessions: dict[str, dict] = {}
sessions_lock = threading.RLock()

_detector: Detector | None = None
_detector_error: str | None = None
_loading = False


# =============================================================== model load
def start_loading():
    """Load all model weights asynchronously at startup (kept non-blocking)."""
    global _loading, _detector, _detector_error

    if _detector is not None or _detector_error or _loading:
        return
    _loading = True

    def _load():
        global _detector, _detector_error, _loading
        try:
            _detector = Detector(CFG)
        except Exception as exc:  # noqa: BLE001 - surfaced via /api/status
            _detector_error = f"{type(exc).__name__}: {exc}"
        finally:
            _loading = False

    threading.Thread(target=_load, daemon=True).start()


def _cleanup():
    """FR-20: expire session files older than the session TTL (1 hour)."""
    now = time.time()
    with sessions_lock:
        expired = [sid for sid, rec in sessions.items()
                   if rec["expires_at"] <= now]
        for sid in expired:
            sessions.pop(sid, None)
            shutil.rmtree(sessions.get(sid, {}).get("dir") or UPLOAD_ROOT / sid,
                          ignore_errors=True)

    if not UPLOAD_ROOT.is_dir():
        return
    for d in list(UPLOAD_ROOT.iterdir()):
        if d.is_dir():
            try:
                stale = time.time() - d.stat().st_mtime > APP_SESSION_EXPIRY
            except OSError:
                stale = False
            if stale:
                shutil.rmtree(d, ignore_errors=True)


def _sweep():
    while True:
        time.sleep(300)
        _cleanup()


def register_sweeper():
    threading.Thread(target=_sweep, daemon=True).start()


# ================================================================ route utils
def _new_session() -> dict:
    sid = uuid.uuid4().hex
    rec = {
        "id": sid,
        "created_at": time.time(),
        "expires_at": time.time() + APP_SESSION_EXPIRY,
        "dir": UPLOAD_ROOT / sid,
    }
    with sessions_lock:
        sessions[sid] = rec
    return rec


def _session_dir(sid: str) -> Path:
    rec = sessions.get(sid)
    if not rec:
        raise KeyError("session")
    return rec["dir"]


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


# ==================================================================== routes
@app.errorhandler(413)
def _payload_too_large(_e):
    return _error("FILE_TOO_LARGE",
                  "File too large. Maximum: 100 MB for videos.", 413)


@app.route("/")
def home():
    return render_template("index.html",
                           max_image_mb=CFG.uploads.max_image_size_mb,
                           max_video_mb=CFG.uploads.max_video_size_mb,
                           expiry_hours=APP_SESSION_EXPIRY // 3600)


@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")


@app.route("/api/status")
def api_status():
    models = []
    if _detector:
        b = _detector.bundle
        models = [n for n, m in (("cnn", b.cnn), ("vit", b.vit), ("lstm", b.lstm))
                  if m is not None]
    return jsonify({
        "ready": _detector is not None,
        "loading": _loading,
        "error": _detector_error,
        "device": str(_detector.device) if _detector else None,
        "models_loaded": models,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """UC-01/UC-02: validate the upload and store it under a session."""
    if _detector is None:
        if _detector_error:
            return _error("MODEL_ERROR", _detector_error, 503)
        return _error("LOADING", "Models are still warming up. Please retry.", 503)

    file = request.files.get("file")
    if not file or not file.filename:
        return _error("NO_FILE", "No file selected.", 400)

    data = file.read()
    try:
        meta = validator.validate(file.filename, data)
    except ValidationError as exc:
        return _error("INVALID_FILE", str(exc), 422)

    rec = _new_session()
    rec["media"] = meta
    dir_ = rec["dir"]
    dir_.mkdir(parents=True, exist_ok=True)

    stem = "".join(c if c.isalnum() or c in "._-" else "_"
                   for c in Path(file.filename).stem)[:80] or "upload"
    stored = f"{stem}.{meta['extension']}"
    (dir_ / stored).write_bytes(data)

    return jsonify({
        "session_id": rec["id"],
        "media": {
            "type": meta["kind"], "extension": meta["extension"],
            "mime": meta["mime"], "size": meta["size"],
            "original_name": file.filename,
        },
        "media_url": f"/media/{rec['id']}/{stored}",
        "expires_in": APP_SESSION_EXPIRY,
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """UC-03/UC-04: run the ensemble pipeline and return the verdict."""
    if _detector is None:
        return _error("NOT_READY", "Detection service is not ready yet.", 503)

    payload = request.get_json(silent=True) or {}
    sid = payload.get("session_id", "")
    rec = sessions.get(sid)
    if not rec:
        return _error("BAD_SESSION", "Upload session expired. Please re-upload.", 404)
    dir_ = rec.get("dir")
    if not dir_ or not dir_.is_dir():
        return _error("BAD_SESSION", "Upload session expired. Please re-upload.", 404)

    media = rec.get("media")
    if not media:
        return _error("NO_MEDIA", "No media uploaded in this session.", 400)

    stored = _find_media(dir_, media)
    if stored is None:
        return _error("NO_MEDIA", "Media file is missing from the session.", 500)

    try:
        if media["kind"] == "video":
            result = _detector.detect_video(stored, dir_,
                                            original_name=media["original_name"])
        else:
            result = _detector.detect_image(stored, dir_,
                                            original_name=media["original_name"])
    except NoFaceError as exc:
        return _error("NO_FACE", str(exc), 422)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("inference failed")
        return _error("INTERNAL",
                      "An internal error occurred. Please try again.", 500)

    result["session_id"] = sid
    result["heatmap_url"] = f"/media/{sid}/{result['heatmap_path']}"
    result["face_url"] = f"/media/{sid}/{result['face_crop_path']}"
    result["media_url"] = f"/media/{sid}/{stored.name}"

    with sessions_lock:
        sessions[sid]["result"] = result

    return jsonify({"ok": True, "result": result})


def _find_media(dir_: Path, media: dict):
    ext = (media.get("extension") or "").lower()
    for f in dir_.iterdir():
        if f.is_file() and f.suffix.lstrip(".").lower() == ext:
            return f
    # fallback: any supported media file in the session dir
    for f in dir_.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpeg", ".jpg", ".png", ".mp4", ".avi"}:
            return f
    return None


@app.route("/media/<sid>/<path:filename>")
def media_file(sid, filename):
    rec = sessions.get(sid)
    if not rec:
        return _error("NOT_FOUND", "Session not found or expired.", 404)
    root = rec["dir"].resolve()
    target = (root / filename).resolve()
    if not str(target).startswith(str(root) + os.sep) and target != root:
        return _error("FORBIDDEN", "Invalid path.", 400)
    if not target.is_file():
        return _error("NOT_FOUND", "File not found.", 404)
    return send_file(target)


@app.route("/metrics-chart/<path:filename>")
def metrics_chart(filename):
    base = METRICS_DIR.resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        return _error("NOT_FOUND", "Not found.", 404)
    return send_file(target)


@app.route("/api/metrics")
def api_metrics():
    import json

    fp = METRICS_DIR / "metrics.json"
    if not fp.is_file():
        return _error("NO_METRICS",
                      "No evaluation results available. Please run model "
                      "evaluation first (UC-07 A1).", 404)
    doc = json.loads(fp.read_text(encoding="utf-8"))
    for key, row in doc.get("models", {}).items():
        row["confusion_url"] = f"/metrics-chart/{row['chart_confusion']}"
        row["roc_url"] = f"/metrics-chart/{row['chart_roc']}"
    return jsonify(doc)


# ================================================================ bootstrap
start_loading()
register_sweeper()

if __name__ == "__main__":
    app.run(host=CFG.server.host, port=int(CFG.server.port),
            debug=bool(CFG.server.debug))