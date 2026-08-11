# DeepFake Detection System

A web application that detects deepfake **images** and **videos** using an
ensemble of four deep-learning models (two retired wrappers are kept on
disk but never loaded):

- **XceptionNet** (CNN) – analyses facial texture/artifacts (images + Grad-CAM)
- **EfficientNet-B3** (CNN) – second spatial backbone, the strongest single
  model in the FF++ C23 study (images + videos)
- **ViT** (CommunityForensics ViT-Small, CVPR 2025) – trained on 2.7M samples
  from 4,803 generators; detects unseen forgery methods (images + videos)
- **ViT-L/14** (LNCLIP-DF, WACV 2026) – CLIP ViT-L/14 with LN-tuning,
  generalizes cross-dataset (FF++ -> Celeb-DF-v2 / DFDC) (videos only)

Retired (files kept, never loaded/voted): dima806 **ViT-B/16** and the
**ResNet18-BiLSTM** temporal model.

Results come with a per-model probability score, a combined verdict
(REAL / FAKE), a confidence %, and a **Grad-CAM heatmap** that shows *which
part of the face* influenced the decision.

---

## 1. What you need

- **Python 3.10 or newer (64-bit)** – download from https://www.python.org
- **Node.js 18+** (optional) – only needed if you rebuild the React UI
- **Internet access once** – to install dependencies and fetch the pre-trained
  model weights (~1.6 GB total). Inference itself is fully offline (NFR-07).
- ~10 GB of **free disk space** to store the model weights and dependencies.

Recommended: **8 GB+ RAM** (CPU inference). A NVIDIA GPU is optional
(`cuda`), but not required.

Supported uploads:

| Type    | Formats        | Max size | Notes                          |
|---------|----------------|----------|--------------------------------|
| Image   | JPEG, PNG      | 10 MB    | Must contain a visible face    |
| Video   | MP4, AVI       | 100 MB   | Only the first 60 s are analysed|

---

## 2. Step-by-step setup (new machine)

### Step 2.1 – Get the code

Open a terminal in the folder where you want the project, then:

```
cd deepfake-detector
```

### Step 2.2 — Create the virtual environment

**Windows (PowerShell / Command Prompt):**

```
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```
python -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your prompt.

### Step 2.3 — Install the dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, OpenCV, Transformers, Flask and the other ~20
libraries the app needs. This can take **5–10 minutes**.

### Step 2.4 — Download the model weights

```
python scripts/download_models.py
```

Downloads every checkpoint into `models/` (the CNN weights, plus the three
transformer model folders). Everything is hosted in the project's own
Hugging Face repository (`Khubaib7/deepfake-models`); files that already
exist locally are skipped, so you can re-run it safely.

Repo layout used by the script (mirror these folders to
`huggingface.co/Khubaib7/deepfake-models`):

```
deepfake-models/
├── xception_weights.pt          -> models/xception_weights.pt
├── efficientnet_weights.pt      -> models/efficientnet_weights.pt
├── ViT/                         -> models/ViT/                    (ViT model)
├── vit_l14/model.torchscript    -> models/vit_l14/model.torchscript (ViT-L/14)
```

> At inference time nothing touches the network (NFR-07): each loader uses
> its local `models/` copy, falling back to `Khubaib7/deepfake-models` on
> the Hub only when the local folder is missing.

---

## 3. Run the app

The UI is a **React** single-page app (built once with Vite into
`frontend/dist/`) served by the Flask backend — so running stays a single
command:

```
python app.py
```

Wait for the line saying the server started, then open:

**http://127.0.0.1:5000** in your browser.

Notes:
- The six models load **in the background**; the page will simply wait
  until they are ready. First start loads ~1.6 GB, allow a minute or two.
- Model loading status can be checked at **http://127.0.0.1:5000/api/status**.

### Building / developing the React UI (optional)

The pre-built UI already lives in `frontend/dist/`, so you don't need Node to
run the app. Only build the frontend after you change the React source:

```
cd frontend
npm install        # first time only
npm run build      # writes frontend/dist (served by Flask on next start)
npm run dev        # dev server on :5173, proxies /api to Flask :5000
```

### How to use the scanner

1. Drag & drop a photo or video onto the drop zone (or click to browse).
2. Click **Detect**.
3. The result shows:
   - The verdict badge: **REAL** or **FAKE**
   - The **probability** of manipulation and the **confidence** %
   - Per-model scores (images: XceptionNet, EfficientNet-B3, ViT; videos:
     EfficientNet-B3, ViT, ViT-L/14. The retired dima806 ViT-B/16 and the
     BiLSTM never vote)
   - A **Grad-CAM heatmap** highlighting manipulated regions
   - The extracted face crop used for analysis

### Developer metrics dashboard

Open **http://127.0.0.1:5000/metrics** to see accuracy / precision / recall /
F1 / ROC-AUC for each model and the combined ensemble, with confusion-matrix
and ROC-curve charts.

---

## 4. Running the tests

```
python tests\test_validator.py      # upload validation (file rules)
python tests\smoke_test.py          # full end-to-end image + video pipeline
```

The smoke test uses the sample files in `tests/fixtures/` and writes its
output to `tests/_smoke_artifacts/`.

---

## 5. Common tasks

| Task                    | Command                                   |
|-------------------------|-------------------------------------------|
| Start the web app       | `python app.py`                           |
| Download model weights | `python scripts/download_models.py`       |
| Rebuild metrics charts  | `python scripts/generate_metrics.py`      |
| Run unit tests          | `python tests/test_validator.py`          |
| Run smoke test          | `python tests/smoke_test.py`              |

Re-run `generate_metrics.py` whenever you change the ensemble weights in
`config.yaml` — the dashboard and live inference always use the same weights.

---

## 6. Configuration

Everything is tunable in **`config.yaml`** without touching code:

| Section          | What it controls                                   |
|------------------|---------------------------------------------------|
| `server`         | Host / port / debug mode                           |
| `uploads`        | Accepted formats, size limits, session expiry      |
| `models`         | Device (`auto`/`cpu`/`cuda`), checkpoints, ViT backend |
| `ensemble`       | Model weights for images & videos, the FAKE threshold |
| `preprocessing`  | Frame rate, maximum frames, face margin, MTCNN confidence |
| `metrics`        | Output folder for charts and metrics JSON          |

---

## 7. Troubleshooting

| Problem                                   | Fix                                                                  |
|-------------------------------------------|----------------------------------------------------------------------|
| `HF_TOKEN warning` at startup             | Harmless. The ViT model is public and downloads without a token.     |
| ViT is slow on first scan                 | First run downloads ~330 MB to `~/.cache/huggingface/hub/`.          |
| "No face detected" error                  | MTCNN requires a clearly visible face (confidence gate 0.95). Try a sharper, front-facing photo. |
| `CNN checkpoint missing / LSTM checkpoint missing` | Run `python scripts/download_models.py`.                        |
| 413 / "File too large"                    | Respect the limits (image 10 MB, video 100 MB) or raise them in `config.yaml` and restart. |
| Slow on CPU                               | Set `models.device: cpu` explicitly, or set to `cuda` if you have a GPU. |

---

For developers: see [README_DEVELOPER.md](README_DEVELOPER.md) for a
file-by-file technical breakdown of the codebase."# deepfake-detector" 
"# deepfake-detector" 
