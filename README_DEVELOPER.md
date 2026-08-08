# DeepFake Detection System — Developer Guide

Technical documentation for maintainers: architecture, the request/data flow,
and a file-by-file breakdown of every module.

---

## 1. Architecture overview

The app is split into three tiers following the SRS (Software Requirements
Specification):

| Tier | Concern | Location |
|------|---------|----------|
| Tier 1 | Web / UI layer (Flask + React) | `app.py`, `frontend/` (built to `frontend/dist/`) |
| Tier 2 | AI engine (core package) | `core/` |
| Tier 3 | Model weights + temporary storage | `models/`, `uploads/` |

The code is annotated with SRS identifiers (e.g. `FR-06`, `UC-03`) so every
requirement can be traced to an implementation.

### Inference flow

```
Browser
  │  POST /api/upload (multipart file)
  ▼
app.py  api_upload ──► core/validator.py   FR-01..FR-04  (magic-byte check,
  │                                                       size limits)
  │  session dir  uploads/<session_id>/    FR-20  (auto-expires in 1 h)
  ▼
app.py  api_detect  (JSON {session_id})
  ▼
core/detector.py  Detector (thread-locked, one inference at a time)
  │
  ├─ IMAGE path (UC-03):
  │    preprocessing.detect_face()  MTCNN ─► aligned 224×224 crop  FR-06..10
  │    models.cnn_xception      → sigmoid P(Fake)
  │    models.cnn_efficientnet  → sigmoid P(Fake)
  │    models.vit_vision        → softmax P(Fake)
  │    gradcam.compute_gradcam_heatmap() → heatmap.png (FR-17)
  │    ensemble.combine(scores, kind="image") → verdict     FR-14..16
  │
  └─ VIDEO path (UC-04):
       preprocessing.extract_video_frames()  ffmpeg 5 fps, first 60 s,
                                              subsample ≤ 24 frames  FR-05
       for each frame: MTCNN face crop (skips no-face frames)
       models.cnn_xception      → per-frame P(Fake), mean
       models.cnn_efficientnet  → per-frame P(Fake), mean
       models.vit_vision        → per-frame P(Fake), mean
       models.lstm_temporal     → P(Fake) across the whole clip  FR-13
       gradcam on the most-manipulated frame
       ensemble.combine(scores, kind="video")                FR-14..16
  │
  ▼
app.py returns DETECTION_RESULT JSON to the browser
```

### Ensemble decision (core/ensemble.py)

Weighted soft-voting (weights from `config.yaml`):

```
P_fake = (w_cnn·P_cnn + w_eff·P_eff + w_vit·P_vit + w_lstm·P_lstm) / Σw
verdict = FAKE if P_fake >= threshold (default 0.5) else REAL
confidence = P_fake·100         if FAKE
             (1 − P_fake)·100   if REAL
```

Images use `cnn + efficientnet + vit`; videos add the LSTM.

---

## 2. File-by-file reference

### Root level

| File | Purpose |
|------|---------|
| `app.py` | Flask entry point (`python app.py`). Defines all HTTP routes, the in-memory session store, the expiry sweeper thread, and the asynchronous model loader. |
| `config.yaml` | Single source of truth for all tunables (upload limits, model checkpoints, ensemble weights, preprocessing). Read by `core/config.py`. |
| `requirements.txt` | Pinned library list (torch, torchvision, transformers, timm, facenet-pytorch, grad-cam, opencv, flask, scikit-learn, matplotlib, seaborn, PyYAML, huggingface_hub). |
| `.gitignore` | Ignores `.venv` and `.env*` files. |
| `models/` | Downloaded checkpoints: `xception_weights.pt`, `vit_ffpp_weights.pth`, `cnn_lstm_weights.pth`. |
| `metrics/` | Offline evaluation artifacts consumed by the dashboard (`metrics.json`, `confusion_*.png`, `roc_*.png`, prediction JSON per model). |
| `uploads/` | Runtime session storage. Each session is one folder; files older than the TTL are deleted by the sweeper. |

### `core/` — Tier 2 AI engine

| File | Purpose |
|------|---------|
| `core/__init__.py` | Package marker. |
| `core/config.py` | Loads `config.yaml` into a `Config` object: `load_config()` → nested attribute access (`cfg.uploads.max_image_size_mb`); `resolve()` resolves relative paths against the project root. Defines `PROJECT_ROOT`. |
| `core/validator.py` | `FileValidator.validate(filename, bytes)` enforces FR-01..FR-04 using **magic-byte sniffing** (`sniff_extension()`), MIME extension cross-check (rejects renamed files, FR-03), and size limits. Raises `ValidationError` with SRS-compliant messages. |
| `core/preprocessing.py` | `FacePreprocessor` wraps MTCNN (from `facenet_pytorch`): detections gated at confidence ≥ 0.95 (`NoFaceError` otherwise, FR-07), largest-face selection, eye-horizontal rotation + margin crop (FR-08), 224×224 resize (FR-09), ImageNet normalisation (FR-10). Also `extract_video_frames()`: FFmpeg (`imageio-ffmpeg` fallback, then OpenCV VideoCapture) extracts frames at configurable fps, first 60 s only, subsampled uniformly to ≤ `max_frames` (FR-05). |
| `core/detector.py` | `Detector` orchestrator. Loads the model bundle, exposes `detect_image()` (UC-03) and `detect_video()` (UC-04), runs Grad-CAM, composes the SRS-compliant result document. Holds a re-entrant lock so model inference is serialised (models are not thread-safe concurrently). |
| `core/ensemble.py` | `weighted_average()`, `decision_from_p()`, and the `Ensemble` class implementing FR-14 (weighted soft voting), FR-15 (threshold verdict), FR-16 (confidence). |
| `core/gradcam.py` | `compute_gradcam_heatmap()` — hooks `pytorch_grad_cam` onto `cnn_model.grad_cam_target_layer` (Xception `conv4`), targets class 0, returns a 0..1 saliency map. `save_heatmap_overlay()` blends the map as a jet colormap over the face crop and writes the PNG (FR-17). |
| `core/metrics.py` | Offline evaluation helpers: `evaluate()` (accuracy/precision/recall/F1/ROC-AUC/confusion-matrix with sklearn), `plot_confusion_matrix()`, `plot_roc_curve()`, `save_metrics_json()`, `load_predictions_json()`. Supports FR-18/FR-19 dashboard data. |

### `core/models/` — model wrappers

| File | Purpose |
|------|---------|
| `core/models/__init__.py` | `get_device()` ('auto' → CUDA if available else CPU), `load_model()` dispatcher, and the `ModelBundle` container (holds .cnn/.effnet/.vit/.lstm + device; `.eval_all()`). |
| `core/models/cnn_xception.py` | `XceptionNet(nn.Module)` wrapping `timm`'s `legacy_xception` backbone with a single logit output. The backbone stays as a named submodule so `grad_cam_target_layer` (= `backbone.conv4`) can be hooked. `load_cnn()` loads the FF++ C23 checkpoint from `models/xception_weights.pt`. |
| `core/models/cnn_efficientnet.py` | `EfficientNetNet(nn.Module)` wrapping `timm`'s `efficientnet_b3` under `.backbone` (FF++ C23 checkpoint with `backbone.`-prefixed keys). The strongest single model in the FF++ study (accuracy 0.9829, AUC 0.9976). |
| `core/models/vit_vision.py` | `HuggingFaceViT` wrapping `AutoModelForImageClassification`. Loads from the local folder `models/vit.checkpoint` (default `./models/ViT`, a self-hosted copy of `dima806/deepfake_vs_real_image_detection`, 99.3% acc); if that folder is absent it falls back to `models.vit.hf_weights` (a HF repo id). P(Fake) = softmax[class 1]. |
| `core/models/lstm_temporal.py` | `TemporalNet` — ResNet-18 feature extractor (fc removed, 512-dim) feeding a 2-layer **bidirectional** LSTM (hidden 256) + MLP head returning one scalar logit for a clip `[T,3,224,224]`. `load_temporal()` loads `models/cnn_lstm_weights.pth`; `clip_logit_to_proba()` maps to P(Fake). |

### `scripts/`

| File | Purpose |
|------|---------|
| `scripts/download_models.py` | Downloads the three FF++ C23 checkpoints from the user's own Hugging Face repo `Khubaib7/deepfake-models` (flat files: `xception_weights.pt`, `cnn_lstm_weights.pth`, `efficientnet_weights.pt`) into the flat `models/` layout. Idempotent (skips existing files). The ViT downloads on first detection run. |
| `scripts/generate_metrics.py` | Reads per-model test predictions, computes the metric set per model **and** the weighted ensemble (weights pulled live from `config.yaml.ensemble.video_weights` so dashboard == inference), regenerates `metrics/metrics.json` plus confusion-matrix and ROC PNGs. |

### `tests/`

| File | Purpose |
|------|---------|
| `tests/test_validator.py` | Table-driven unit tests for upload rules (FR-01..FR-04): accepts JPEG/PNG/MP4/AVI, rejects GIF, magic-byte/extension mismatch, oversized files, empty file, and the exact size boundary. |
| `tests/smoke_test.py` | End-to-end smoke test: validates fixtures, runs the full image pipeline (UC-03) and video pipeline (UC-04) through a real `Detector`, asserts on result shape, per-model scores, and artifact PNGs being written. |

### `frontend/` (React UI)

| File | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Landing page + lightweight router (`/` vs `/metrics`), originally ported from the old `templates/index.html` + `static/js/app.js` (since removed). |
| `frontend/src/components/Scanner.jsx` | Scanner in React state/hooks: readiness polling (`/api/status`), drag & drop, client-side size/type checks, upload → detect → result rendering, Grad-CAM/face/source tabs, animated verdict. |
| `frontend/src/pages/Metrics.jsx` | Dashboard in React (UC-07): fetches `/api/metrics`, per-model tabs, stat cards with SRS-target pass/fail, confusion-matrix & ROC images. |
| `frontend/src/style.css` | The single stylesheet (was `static/css/style.css`, now lives in the frontend). |
| `frontend/vite.config.js` | Builds to `frontend/dist/`; dev server on `:5173` proxying `/api` + `/media` → Flask `:5000`. |
| `frontend/dist/` | Compiled React app (index.html + hashed assets), **served by Flask** when present (`/` and `/metrics`); a 503 hints to run `npm run build` otherwise. |

---

## 3. HTTP API reference

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Scanner UI — serves `frontend/dist/index.html` (React). 503 if the frontend wasn't built. |
| GET | `/metrics` | Developer dashboard (React, same bundle). |
| GET | `/assets/<file>` | Compiled React JS/CSS bundles. |
| GET | `/api/config` | Upload limits for the client → `{maxImageMb, maxVideoMb, expiryHours}`. |
| GET | `/api/status` | Readiness probe → `{ready, loading, error, device, models_loaded}`. |
| POST | `/api/upload` | Multipart `file` → validates + stores, returns `session_id`, `media` metadata, `media_url`, `expires_in`. 413/422/503 on failures. |
| POST | `/api/detect` | JSON `{session_id}` → runs pipeline → returns `{ok, result}` where `result` = verdict, p_fake, confidence, threshold, per-model scores (cnn/effnet/vit/lstm), `heatmap_url`, `face_url`, `media_url`, video meta. |
| GET | `/media/<sid>/<file>` | Serves session-scoped artifacts with **path-traversal protection** (`resolve()` + prefix check). |
| GET | `/metrics-chart/<name>` | Serves PNG charts from `metrics/`. |
| GET | `/api/metrics` | JSON metrics document (`metrics/metrics.json`), enriched with `confusion_url`/`roc_url`. 404 if you haven’t run `generate_metrics.py`. |

---

## 4. Key implementation details

- **Asynchronous model load** — `app.py:start_loading()` spins a daemon
  thread so the server is reachable while ~1.6 GB of weights load. Failures
  surface via `/api/status` (`error` field) and uploads are rejected until ready.
- **Session lifecycle** — each upload gets a UUID folder under `uploads/sid`.
  A 5-minute sweeper thread (`register_sweeper()` → `_cleanup()`) deletes
  sessions whose TTL (1 h) elapsed, both from the in-memory registry and disk.
- **Concurrency safety** — `Detector` holds a `threading.RLock`; MTCNN runs on
  CPU, and forward passes are wrapped in `torch.no_grad()` / `inference_mode()`
  (Grad-CAM deliberately runs *outside* `inference_mode` because it needs autograd).
- **FFmpeg fallback chain** — frame extraction tries system `ffmpeg` → the
  binary bundled with `imageio-ffmpeg` → pure OpenCV `VideoCapture`. Fully
  local (NFR-07); nothing is sent to a remote service.
- **Input sniffing** — validation reads the first 64 bytes only; MP4 detection
  inspects the `ftyp` brand at offsets 4..12.

---

## 5. Config reference (config.yaml)

| Section | Key | Default | Meaning |
|---------|-----|---------|---------|
| `server` | `host` / `port` / `debug` | `127.0.0.1` / `5000` / `false` | Flask binding |
| `uploads` | `max_image_size_mb` / `max_video_size_mb` | 10 / 100 | FR-04 limits |
| `uploads` | `allowed_image_types` / `allowed_video_types` | jpeg,png / mp4,avi | FR-01/FR-02 |
| `uploads` | `max_video_duration_seconds` | 60 | FR-02 |
| `uploads` | `session_expiry_seconds` | 3600 | FR-20 sweep |
| `models` | `device` | `auto` | `auto`/`cpu`/`cuda` |
| `models.vit` | `checkpoint` / `hf_repo` + `hf_subfolder` | `./models/ViT` | Local folder with the model files (primary, no network); if absent, falls back to `hf_repo` (e.g. `Khubaib7/deepfake-models`) + `hf_subfolder` (`ViT`) on the owner's account |
| `ensemble` | `image_weights` / `video_weights` | all 1.0 | Soft-vote weights (cnn/effnet/vit [/lstm]) |
| `ensemble` | `threshold` | 0.5 | FAKE cutoff |
| `preprocessing` | `video_fps` / `max_frames` | 5 / 24 | frame extraction |
| `preprocessing` | `face_margin` / `mtcnn_confidence` | 0.30 / 0.95 | crop pad / conf gate |
| `preprocessing` | `target_size` | 224 | model input size (FR-09) |
| `preprocessing` | `imagenet_mean` / `imagenet_std` | standard values | FR-10 |

---

## 6. Reproducing the metrics

1. `python scripts/download_models.py` — fetch per-model test predictions.
2. `python scripts/generate_metrics.py` — recompute + plot charts, writes
   `metrics/metrics.json`.

SRS quality targets used by the dashboard (NFR-03/04/05): accuracy ≥ 0.75,
ROC-AUC ≥ 0.85, F1 ≥ 0.80. Current FF++ C23 numbers (from `metrics.json`):
Xception 97.4% / AUC 0.994, EfficientNet-B3 98.3% / 0.998, LSTM 97.4% / 0.980,
ensemble 98.0% / 0.999. (The runtime ViT — dima806, 99.3% acc — has no
released FF++ C23 test predictions, so it is excluded from the offline
dashboard numbers but still contributes at inference time.)