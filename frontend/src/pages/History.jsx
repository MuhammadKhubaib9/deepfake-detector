import { useEffect, useState } from "react";

const humanSize = null; // sizes are stored relative to session url; not needed here
const fmtDate = iso => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit"
  });
};

export default function History() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [filterKind, setFilterKind] = useState("");
  const [filterVerdict, setFilterVerdict] = useState("");
  const [selected, setSelected] = useState(null);

  function load() {
    setLoading(true);
    setError("");
    const q = new URLSearchParams();
    if (filterKind) q.set("kind", filterKind);
    if (filterVerdict) q.set("verdict", filterVerdict);
    q.set("limit", "200");
    fetch("/api/history?" + q.toString())
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(d => {
        setItems(d.items || []);
        setTotal(d.total || 0);
        if (selected && !d.items.some(x => x.id === selected)) setSelected(null);
      })
      .catch(err => setError((err && err.error && err.error.message) || "Failed to load history."))
      .finally(() => setLoading(false));
  }

  useEffect(load, [filterKind, filterVerdict]);

  function remove(id) {
    fetch("/api/history/" + id, { method: "DELETE" })
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(() => { if (selected === id) setSelected(null); load(); })
      .catch(err => setError((err && err.error && err.error.message) || "Delete failed."));
  }

  function clearAll() {
    if (!confirm("Delete all scan history?")) return;
    fetch("/api/history", { method: "DELETE" })
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(() => { setSelected(null); load(); })
      .catch(err => setError((err && err.error && err.error.message) || "Clear failed."));
  }

  const detailItem = selected != null ? items.find(x => x.id === selected) : null;

  return (
    <div>
      <header className="topbar">
        <div className="brand">
          <span className="logo">🗂️</span>
          <div>
            <h1>Detection History</h1>
            <p className="tagline">Past scans stored locally · {total} total</p>
          </div>
        </div>
        <nav>
          <a href="/" className="btn ghost">← Back to Detector</a>
        </nav>
      </header>

      <main className="container">
        {error && <div className="alert error">{error}</div>}

        <div className="hist-toolbar">
          <select value={filterKind} onChange={e => setFilterKind(e.target.value)} aria-label="Filter by type">
            <option value="">All media types</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
          </select>
          <select value={filterVerdict} onChange={e => setFilterVerdict(e.target.value)} aria-label="Filter by verdict">
            <option value="">All verdicts</option>
            <option value="FAKE">FAKE</option>
            <option value="REAL">REAL</option>
          </select>
          <button className="btn ghost small" disabled={!total} onClick={clearAll}>Clear all</button>
        </div>

        {loading && <p className="muted">Loading history…</p>}

        {!loading && items.length === 0 && (
          <div className="panel empty-panel">
            <p>No scans recorded yet. Run a detection from the scanner and it will appear here.</p>
          </div>
        )}

        {!loading && items.length > 0 && (
          <div className="hist-grid">
            <div className="panel hist-list">
              {items.map(r => (
                <div key={r.id}
                  className={"hist-row" + (selected === r.id ? " active" : "")}
                  onClick={() => setSelected(r.id)}>
                  <span className={"badge-mini " + r.verdict}>{r.verdict}</span>
                  <div className="hist-row-body">
                    <div className="hist-name">{r.original_name}</div>
                    <div className="hist-meta">
                      {r.kind === "video" ? "🎞️ video" : "🖼️ image"}
                      {" · " + fmtDate(r.created_at)}
                      {r.frames_analyzed ? " · " + r.frames_analyzed + " frames" : ""}
                    </div>
                  </div>
                  <button className="btn ghost small" onClick={e => { e.stopPropagation(); remove(r.id); }}>✕</button>
                </div>
              ))}
            </div>

            {detailItem && (
              <div className="hist-detail panel">
                <div className="hist-detail-head">
                  <span className={"badge " + detailItem.verdict}>{detailItem.verdict}</span>
                  <h3>{detailItem.original_name}</h3>
                  <span className="muted">{fmtDate(detailItem.created_at)}</span>
                </div>
                <div className="hist-media">
                  {detailItem.heatmap_url && <img src={detailItem.heatmap_url} alt="Heatmap" />}
                  {!detailItem.heatmap_url && detailItem.media_url && (
                    <img src={detailItem.media_url} alt="Source" />
                  )}
                </div>
                <div className="meta">
                  <b>{detailItem.kind === "video" ? "Video" : "Image"}</b>
                  {" · confidence " + detailItem.confidence + "%"}
                  {detailItem.frames_analyzed ? " · frames " + detailItem.frames_analyzed : ""}
                </div>
                <div className="scores">
                  {Object.entries(detailItem.scores || {}).map(([k, v]) => (
                    <span className="chip" key={k}>{k}: {(+v).toFixed(2)}</span>
                  ))}
                </div>
                <button className="btn secondary small" onClick={() => remove(detailItem.id)}>Delete this scan</button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer"><small>History stored locally in SQLite · data/history.db</small></footer>
    </div>
  );
}