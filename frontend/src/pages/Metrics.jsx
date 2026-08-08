import { useEffect, useState } from "react";

const SRS_ROWS = [
  ["Accuracy", "accuracy", "accuracy"],
  ["Precision", "precision", null],
  ["Recall", "recall", null],
  ["F1-Score", "f1", "f1"],
  ["ROC-AUC", "roc_auc", "roc_auc"]
];

export default function Metrics() {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState("");
  const [active, setActive] = useState(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(d => {
        setDoc(d);
        const keys = Object.keys(d.models || {});
        if (keys.length) setActive(keys[0]);
      })
      .catch(err => setError((err && err.error && err.error.message) || "Failed to load metrics. Run: python scripts/generate_metrics.py"));
  }, []);

  const models = (doc && doc.models) || {};
  const order = Object.keys(models);
  const m = models[active];
  const srs = (doc && doc.srs_targets) || {};

  return (
    <div>
      <header className="topbar">
        <div className="brand">
          <span className="logo">📊</span>
          <div>
            <h1>Developer Metrics Dashboard</h1>
            <p className="tagline">UC-07 · Confusion matrix, ROC curves &amp; FR-18/FR-19 model metrics</p>
          </div>
        </div>
        <nav><a href="/" className="btn ghost">← Back to Detector</a></nav>
      </header>

      <main className="container">
        {error && <div className="alert error">{error}</div>}

        <div className="tabs" id="model-tabs">
          {order.map(key => (
            <button key={key} className={"tab" + (key === active ? " active" : "")} onClick={() => setActive(key)}>
              {models[key].model_name}
            </button>
          ))}
        </div>

        <section className="stat-grid">
          {m && SRS_ROWS.map(([label, key, targetKey]) => {
            const v = m.metrics[key];
            const target = targetKey ? srs[targetKey] : null;
            return (
              <div className="stat" key={key}>
                <div className="k">{label}</div>
                <div className="v">{v != null ? (v * 100).toFixed(1) + "%" : "—"}</div>
                {target != null && v != null && (
                  <div className={v >= target ? "pass" : "fail"}>
                    {v >= target ? "✓ ≥ SRS target" : "✗ below " + (target * 100).toFixed(0) + "%"}
                  </div>
                )}
              </div>
            );
          })}
        </section>

        <div className="chart-row">
          <figure className={"chart" + (m && m.confusion_url ? "" : " hidden")}>
            <img src={m && m.confusion_url} alt="Confusion matrix" />
            <figcaption>Confusion matrix heatmap</figcaption>
          </figure>
          <figure className={"chart" + (m && m.roc_url ? "" : " hidden")}>
            <img src={m && m.roc_url} alt="ROC curve" />
            <figcaption>ROC curve</figcaption>
          </figure>
        </div>

        <p className="dash-note">{(doc && doc.note) || ""} {m ? "Dataset: " + (m.dataset || "—") : ""}</p>
      </main>

      <footer className="footer"><small>Maps to FR-18/FR-19 &amp; UC-07 (View Metrics Dashboard)</small></footer>
    </div>
  );
}