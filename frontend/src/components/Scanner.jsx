import { useEffect, useState } from "react";

const humanSize = b => b >= 1048576 ? (b / 1048576).toFixed(2) + " MB" : (b / 1024).toFixed(1) + " KB";
const isImage = name => /\.(jpe?g|png)$/i.test(name || "");
const isSupported = name => /\.(jpe?g|png|mp4|avi)$/i.test(name || "");
const TAB_LABELS = { source: "Source", face: "Detected face", heatmap: "Heatmap", verdict: "Verdict" };

const STEPS = [
  ["Detecting faces…", "MTCNN face localization", "face", "face"],
  ["Running deep learning models…", "CNN + EfficientNet + ViT + BiLSTM ensemble", "heatmap", "heatmap"],
  ["Generating Grad-CAM proof…", "Explainability heatmap", "verdict", "verdict"]
];

export default function Scanner() {
  const [file, setFile] = useState(null);
  const [uploadType, setUploadType] = useState(null);
  const [urls, setUrls] = useState({});
  const [sessionId, setSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [error, setError] = useState("");
  const [scanState, setScanState] = useState("Waiting for a file");
  const [scanning, setScanning] = useState(false);
  const [loader, setLoader] = useState(null);
  const [ready, setReady] = useState(false);
  const [device, setDevice] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [, setRenderTick] = useState(0);

  useEffect(() => {
    fetch("/api/config")
      .then(r => r.json())
      .then(cfg => { window.DF_CONFIG = cfg; })
      .catch(() => {});
  }, []);

  useEffect(() => {
    let alive = true;
    (function poll() {
      fetch("/api/status")
        .then(r => r.json())
        .then(s => {
          if (!alive) return;
          setDevice(s.requested_device || s.device);
          if (s.ready) { setReady(true); setError(prev => prev === "LOADING" ? "" : prev); }
          else if (s.error) setError("Model load failed: " + s.error);
          else {
            setScanState(s.loading_stage ? "Loading: " + s.loading_stage : "Waiting for a file");
            setTimeout(poll, 2000);
          }
        })
        .catch(() => setTimeout(poll, 3000));
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (file && !verdict) setActiveTab(null);
  }, [file, verdict]);

  function pickFile(f, type) {
    setError("");
    setVerdict(null);
    setActiveTab(null);
    if (!f) {
      const inp = document.getElementById("file-input");
      if (inp) {
        inp.accept = type === "image" ? ".jpeg,.jpg,.png" : ".mp4,.avi";
        inp.click();
      }
      return;
    }
    if (!isSupported(f.name)) {
      setError("Unsupported format. Please use JPEG, PNG, MP4 or AVI.");
      return;
    }
    const isImg = isImage(f.name);
    if (type && isImg !== (type === "image")) {
      setError(type === "image"
        ? "You selected an image upload but that's not an image. Pick a video file."
        : "You selected a video upload but that's not a video. Pick an image file.");
      return;
    }
    const maxMB = isImg
      ? (window.DF_CONFIG && window.DF_CONFIG.maxImageMb) || 10
      : (window.DF_CONFIG && window.DF_CONFIG.maxVideoMb) || 100;
    if (f.size > maxMB * 1048576) {
      setError(isImg
        ? "File too large. Maximum: 10 MB for images."
        : "File too large. Maximum: 100 MB for videos.");
      return;
    }
    setFile(f);
    setUploadType(isImg ? "image" : "video");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setScanState("Ready to scan");
  }

  function showError(msg) {
    setError(msg);
    setScanState("Failed");
  }

  const [switching, setSwitching] = useState(false);

  async function switchDevice(target) {
    if (switching) return;
    setSwitching(true);
    setError("");
    try {
      const res = await fetch("/api/device", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: target })
      });
      const d = await res.json();
      if (!res.ok) throw d;
      setDevice(target === "auto" ? (d.device) : target);
      setScanState("Switching device — reloading models…");
    } catch (err) {
      setError((err && err.error && err.error.message) || "Device switch failed.");
    } finally {
      setSwitching(false);
    }
  }

  async function detect() {
    if (!file || scanning) return;
    setScanning(true);
    setError("");
    setScanState("Scanning…");
    setLoader({ msg: "Analyzing media…", step: "Uploading" });
    setActiveTab("source");
    let i = 0;
    const timer = setInterval(() => {
      if (i >= STEPS.length) return;
      const s = STEPS[i];
      setLoader({ msg: s[0], step: s[1] });
      setActiveTab(s[2]);
      i += 1;
    }, 2600);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const upRes = await fetch("/api/upload", { method: "POST", body: fd });
      const up = await upRes.json();
      if (!upRes.ok) throw up;
      setSessionId(up.session_id);
      const detRes = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: up.session_id })
      });
      const det = await detRes.json();
      if (!detRes.ok) throw det;
      renderResult(det.result);
    } catch (err) {
      setError((err && err.error && err.error.message) || "An internal error occurred. Please try again.");
      setScanState("Failed");
    } finally {
      clearInterval(timer);
      setLoader(null);
      setScanning(false);
    }
  }

  function renderResult(res) {
    setVerdict(res);
    setUrls({
      source: res.media_url,
      face: res.face_url,
      heatmap: res.heatmap_url
    });
    setActiveTab("verdict");
    setScanState("Scan complete");
    animateConfidence(res.confidence);
  }

  function animateConfidence(target) {
    const start = performance.now();
    const dur = 1100;
    (function tick(ts) {
      const p = Math.min((ts - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setConfidence((target * eased).toFixed(1));
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }

  function reset() {
    setVerdict(null);
    setUrls({});
    setFile(null);
    setUploadType(null);
    setSessionId(null);
    setActiveTab(null);
    setError("");
    setConfidence(null);
    setPreviewUrl("");
    setScanState("Waiting for a file");
    const inp = document.getElementById("file-input");
    if (inp) inp.value = "";
  }

  const FILE_SENT = file && !verdict;
  const showStage = FILE_SENT;
  const showTabs = verdict;
  const showVerdict = verdict && activeTab === "verdict";
  const previewImg = file && previewUrl && isImage(file.name);

  return (
    <div className="scanner card">
      <div className="scanner-head">
        <span className="pulse-dot"></span>
        <span className="sc-title">DeepFake Scan</span>
        <span className="device-switch" title="Switch inference device">
          <button type="button" className={"dev-btn" + (device === "cpu" ? " active" : "")}
            disabled={scanning || switching}
            onClick={() => switchDevice("cpu")}>CPU</button>
          <button type="button" className={"dev-btn" + (device === "cuda" ? " active" : "")}
            disabled={scanning || switching}
            onClick={() => switchDevice("cuda")}>GPU</button>
        </span>
        <span className="sc-status" style={scanState === "Scanning…" ? { color: "var(--accent)" } : undefined}>{scanState}</span>
      </div>

      {!file && (
        <div className="upload-picker">
          <input type="file" id="file-input"
            accept=".jpeg,.jpg,.png,.mp4,.avi" hidden
            onChange={e => pickFile(e.target.files[0])} />
          <div className="pick-row">
            <div className="pick-card" role="button" tabIndex={0}
              onClick={() => pickFile(null, "image")}
              onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(null, "image"); } }}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                if (e.dataTransfer.files.length) {
                  const f = e.dataTransfer.files[0];
                  if (!isImage(f.name)) { setError("This isn't an image file. Drag a JPEG or PNG here."); return; }
                  pickFile(f, "image");
                }
              }}>
              <div className="pick-ico">🖼️</div>
              <p className="pick-title">Upload an image</p>
              <p className="pick-sub">or <span className="link">click to browse</span></p>
              <div className="dz-formats"><span className="fchip">JPEG</span><span className="fchip">PNG</span></div>
            </div>
            <div className="pick-card" role="button" tabIndex={0}
              onClick={() => pickFile(null, "video")}
              onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(null, "video"); } }}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                if (e.dataTransfer.files.length) {
                  const f = e.dataTransfer.files[0];
                  if (isImage(f.name)) { setError("Only video files can be dropped in the video section."); return; }
                  pickFile(f, "video");
                }
              }}>
              <div className="pick-ico">🎞️</div>
              <p className="pick-title">Upload a video</p>
              <p className="pick-sub">or <span className="link">click to browse</span></p>
              <div className="dz-formats"><span className="fchip">MP4</span><span className="fchip">AVI</span></div>
            </div>
          </div>
        </div>
      )}

      {showStage && (
        <div className="stage">
          <div className="stage-media">
            {previewImg && <img src={previewUrl} alt="Media under analysis" />}
            {!previewImg && <video src={previewUrl} controls></video>}
            <div className="scan-sweep"></div>
          </div>
          <div className="preview-meta">
            <span className="chip">{file.name}</span>
            <span className="chip">{humanSize(file.size)}</span>
            <button className="btn small ghost" type="button" onClick={() => pickFile(null, uploadType)}>↻ Replace</button>
          </div>
          <button className="btn primary btn-block" disabled={scanning} onClick={detect}>
            <span className="btn-ico">🎯</span> Analyze for deepfakes
          </button>
        </div>
      )}

      {showTabs && (
        <>
          <div className="tabs-row">
            {["source", "face", "heatmap", "verdict"].map(t => (
              <button key={t} className={"stab" + (activeTab === t ? " active" : "")}
                onClick={() => setActiveTab(t)} disabled={scanning}>
                {TAB_LABELS[t]}
              </button>
            ))}
          </div>

          {!showVerdict && urls[activeTab] && (
            <div className="stage">
              <div className="stage-media">
                {isImage(urls[activeTab])
                  ? <img src={urls[activeTab]} alt={activeTab} />
                  : <video src={urls[activeTab]} controls></video>}
              </div>
            </div>
          )}

          {showVerdict && (
            <VerdictView verdict={verdict} onAgain={reset} />
          )}
        </>
      )}

      {error && <div className="alert error">{error}</div>}
      {loader && (
        <div className="loader">
          <div className="scan-frame"><span className="scanline"></span></div>
          <p>{loader.msg}</p>
          <p className="loader-step">{loader.step}</p>
        </div>
      )}
    </div>
  );
}

function VerdictView({ verdict, onAgain }) {
  const isFake = verdict.verdict === "FAKE";
  return (
    <div className="verdict-view">
      <div className="verdict-wrap">
        <span className={"verdict-ring pop-" + (isFake ? "fake" : "real")}></span>
        <div className={"badge " + verdict.verdict + " show"}>{verdict.verdict}</div>
      </div>
      <div className="quick">
        <p className="result-note" dangerouslySetInnerHTML={{
          __html: isFake
            ? "This media shows strong signs of <b>AI manipulation</b>. Please verify before trusting it."
            : "No significant signs of manipulation found. This media appears <b>authentic</b>."
        }}></p>
        {verdict.kind === "video" && (
          <div className="meta">
            Frames analyzed: <b>{verdict.faces_analyzed}</b>
            {verdict.most_manipulated_frame ? " · Most suspicious frame: #" + verdict.most_manipulated_frame : ""}
            {verdict.video_fps ? " · Extraction rate: " + verdict.video_fps + " fps" : ""}
          </div>
        )}
        <div className="actions">
          <button className="btn secondary" onClick={onAgain}>Scan another file</button>
          <span className="meta-text">Analyzed the detected face only · frames: {verdict.faces_analyzed || 1}</span>
        </div>
      </div>
    </div>
  );
}