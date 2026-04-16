import sqlite3

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from utils.logger import LOG_DIR

ui_router = APIRouter()


@ui_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse(content=_HTML)


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Detection Service</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh; padding: 30px 20px; }
  h1  { text-align: center; font-size: 1.6rem; margin-bottom: 6px; color: #fff; }
  .sub { text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 28px; }

  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
           padding: 24px; max-width: 860px; margin: 0 auto 24px; }

  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 18px; }

  label { display: block; font-size: 0.78rem; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }

  .drop-zone { flex: 1; min-width: 220px; border: 2px dashed #333; border-radius: 8px;
               padding: 22px; text-align: center; cursor: pointer; transition: border-color .2s;
               background: #111; }
  .drop-zone:hover, .drop-zone.over { border-color: #4ade80; }
  .drop-zone input { display: none; }
  .drop-zone .icon { font-size: 2rem; margin-bottom: 6px; }
  .drop-zone .hint { font-size: 0.82rem; color: #666; }

  select, .btn {
    padding: 10px 18px; border-radius: 8px; border: 1px solid #333;
    font-size: 0.9rem; cursor: pointer;
  }
  select { background: #111; color: #e0e0e0; }
  .btn { background: #4ade80; color: #000; font-weight: 700;
         border-color: #4ade80; transition: background .15s; }
  .btn:hover { background: #22c55e; }
  .btn:disabled { background: #2a2a2a; color: #555; border-color: #2a2a2a; cursor: not-allowed; }

  .preview-wrap { display: flex; gap: 20px; flex-wrap: wrap; }
  .preview-box  { flex: 1; min-width: 260px; }
  .preview-box img { width: 100%; border-radius: 8px; border: 1px solid #2a2a2a; display: none; }
  .preview-box .caption { font-size: 0.75rem; color: #666; margin-top: 6px; text-align: center; }

  .stats { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
  .stat  { background: #111; border: 1px solid #2a2a2a; border-radius: 8px;
            padding: 10px 16px; flex: 1; min-width: 120px; }
  .stat .val { font-size: 1.5rem; font-weight: 700; color: #4ade80; }
  .stat .lbl { font-size: 0.72rem; color: #666; text-transform: uppercase; }

  .face-list { display: flex; flex-direction: column; gap: 8px; }
  .face-item { background: #111; border: 1px solid #2a2a2a; border-radius: 8px;
                padding: 10px 14px; font-size: 0.83rem; display: flex; gap: 16px; flex-wrap: wrap; }
  .face-item .conf { color: #4ade80; font-weight: 700; }
  .face-item .bbox { color: #888; }

  .spinner { display: none; text-align: center; padding: 20px; color: #888; }
  .error   { background: #2a1010; border: 1px solid #ff4444; border-radius: 8px;
              padding: 12px 16px; color: #ff8888; display: none; font-size: 0.85rem; }

  .log-field { background:#0d0d0d;border:1px solid #222;border-radius:6px;padding:8px 12px; }
  .log-key   { display:block;font-size:.68rem;color:#666;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:3px; }
  .log-val   { color:#e0e0e0;font-size:.82rem;word-break:break-all; }
  .log-val.mono { font-family:monospace;color:#4ade80;font-size:.75rem; }

  #results { display: none; }
</style>
</head>
<body>

<h1>Face Detection Service</h1>
<p class="sub">Upload an image and select a detector to see results</p>

<div class="nav" style="display:flex;justify-content:center;gap:20px;margin-bottom:20px;">
  <a href="/ui" style="color:#4ade80;text-decoration:none;font-size:.85rem;">Detection</a>
  <a href="/ui/live" style="color:#888;text-decoration:none;font-size:.85rem;">Live Camera</a>
  <a href="/ui/logs" style="color:#888;text-decoration:none;font-size:.85rem;">Logs</a>
  <a href="/docs" style="color:#888;text-decoration:none;font-size:.85rem;">API Docs</a>
</div>

<div class="card">
  <div class="row">
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/*">
      <div class="icon" style="font-size:2rem;color:#666">+</div>
      <div id="dropLabel">Drop image here or click to browse</div>
      <div class="hint">JPEG · PNG · BMP · WEBP</div>
    </div>

    <div>
      <label>Detector</label>
      <select id="detector">
        <option value="haar">Haar Cascade</option>
        <option value="mediapipe" selected>MediaPipe</option>
        <option value="yolo">YOLOv8-face</option>
      </select>
    </div>

    <div>
      <label>Min confidence: <span id="confVal">30%</span></label>
      <input type="range" id="minConf" min="0" max="1" step="0.05" value="0.3"
             style="width:120px;accent-color:#4ade80;vertical-align:middle"
             oninput="document.getElementById('confVal').textContent=Math.round(this.value*100)+'%'">
    </div>

    <div>
      <label>User ID (optional)</label>
      <input id="userId" type="text" placeholder="e.g. alice"
             style="padding:9px 12px;background:#111;border:1px solid #333;border-radius:8px;
                    color:#e0e0e0;font-size:.9rem;width:130px">
    </div>

    <button class="btn" id="detectBtn" disabled onclick="runDetection()">Detect Faces</button>
  </div>

  <div style="margin-top:14px">
    <label style="margin-bottom:8px">Or try a sample</label>
    <div id="sampleGallery" style="display:flex;gap:8px;flex-wrap:wrap"></div>
  </div>

  <div class="error" id="errorBox"></div>
  <div class="spinner" id="spinner">Running detection...</div>

  <div id="results">
    <div class="stats">
      <div class="stat"><div class="val" id="statFaces">—</div><div class="lbl">Faces found</div></div>
      <div class="stat"><div class="val" id="statTime">—</div><div class="lbl">Time (ms)</div></div>
      <div class="stat"><div class="val" id="statDetector">—</div><div class="lbl">Detector</div></div>
      <div class="stat"><div class="val" id="statSize">—</div><div class="lbl">Image size</div></div>
    </div>

    <div class="preview-wrap" style="margin-bottom:18px">
      <div class="preview-box">
        <label>Original</label>
        <img id="origImg" src="">
        <div class="caption">Source image</div>
      </div>
      <div class="preview-box">
        <label>Annotated</label>
        <img id="annotImg" src="">
        <div class="caption">Detected faces with bounding boxes</div>
      </div>
    </div>

    <label>Detected Faces</label>
    <div class="face-list" id="faceList"></div>

    <div style="margin-top:20px">
      <label style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
        Request Log Entry
        <span style="font-size:.72rem;color:#555;font-weight:normal;text-transform:none;letter-spacing:0">
          — all fields below are stored in SQLite + JSON log file
        </span>
      </label>
      <div id="logPanel" style="background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:14px"></div>
    </div>
  </div>
</div>

<script>
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});
fileInput.addEventListener('change', () => fileInput.files[0] && setFile(fileInput.files[0]));

function setFile(f) {
  selectedFile = f;
  document.getElementById('dropLabel').textContent = f.name;
  detectBtn.disabled = false;
}

const SAMPLE_FILES = [
  'Adam Driver1.webp', 'Adam Sandler.avif', 'Adrien Brody.jpg',
  'Anthony Hopkins.jpg', 'Antonio Banderas.jpg', 'Ben Affleck.webp',
  'Benedict Cumberbatch1.jpg', 'Bill Murray1.jpg', 'Billy Crystal.jpg',
  'BradPitt1.jpg',
];

(function buildGallery() {
  const gal = document.getElementById('sampleGallery');
  SAMPLE_FILES.forEach(name => {
    const url = '/samples/' + encodeURIComponent(name);
    const img = document.createElement('img');
    img.src = url;
    img.title = name.replace(/\.[^.]+$/, '');
    img.style.cssText = 'width:64px;height:64px;object-fit:cover;border:1px solid #333;border-radius:6px;cursor:pointer;transition:border-color .15s';
    img.onmouseover = () => img.style.borderColor = '#4ade80';
    img.onmouseout  = () => img.style.borderColor = '#333';
    img.onclick = async () => {
      const resp = await fetch(url);
      const blob = await resp.blob();
      const file = new File([blob], name, { type: blob.type });
      setFile(file);
      runDetection();
    };
    gal.appendChild(img);
  });
})();

async function runDetection() {
  if (!selectedFile) return;
  const detector = document.getElementById('detector').value;

  document.getElementById('errorBox').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('spinner').style.display = 'block';
  detectBtn.disabled = true;

  // Show original image immediately
  const origUrl = URL.createObjectURL(selectedFile);
  const origImg = document.getElementById('origImg');
  origImg.src = origUrl;
  origImg.style.display = 'block';

  const minConf  = document.getElementById('minConf').value;
  const userId   = document.getElementById('userId').value.trim();
  const reqStart = new Date().toISOString();
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('detector', detector);
  form.append('return_annotated', 'true');
  form.append('min_confidence', minConf);
  if (userId) form.append('user_id', userId);

  try {
    const resp = await fetch('/detect', { method: 'POST', body: form });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || 'Detection failed');
      return;
    }

    // Stats
    document.getElementById('statFaces').textContent    = data.face_count;
    document.getElementById('statTime').textContent     = data.processing_time_ms.toFixed(1);
    document.getElementById('statDetector').textContent = data.detector;
    document.getElementById('statSize').textContent     = data.image_width + '×' + data.image_height;

    // Annotated image
    const annotImg = document.getElementById('annotImg');
    if (data.annotated_image_base64) {
      annotImg.src = 'data:image/png;base64,' + data.annotated_image_base64;
      annotImg.style.display = 'block';
    }

    // Face list
    const list = document.getElementById('faceList');
    list.innerHTML = '';
    if (data.faces.length === 0) {
      list.innerHTML = '<div style="color:#666;font-size:.85rem;padding:10px">No faces detected in this image.</div>';
    } else {
      data.faces.forEach((f, i) => {
        const b = f.bbox;
        const lms = f.landmarks ? f.landmarks.length + ' landmarks' : 'no landmarks';
        list.innerHTML += `
          <div class="face-item">
            <span>#${i+1}</span>
            <span class="conf">Confidence: ${(f.confidence * 100).toFixed(1)}%</span>
            <span class="bbox">Box: (${b.x1}, ${b.y1}) → (${b.x2}, ${b.y2})</span>
            <span class="bbox">${lms}</span>
          </div>`;
      });
    }

    // ── Log entry panel — all 7 required fields ──────────────────────
    document.getElementById('logPanel').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px">
        <div class="log-field">
          <span class="log-key">① Request ID</span>
          <span class="log-val mono">${data.request_id}</span>
        </div>
        <div class="log-field">
          <span class="log-key">② Timestamp</span>
          <span class="log-val">${reqStart}</span>
        </div>
        <div class="log-field">
          <span class="log-key">③ Detector used</span>
          <span class="log-val">${data.detector}</span>
        </div>
        <div class="log-field">
          <span class="log-key">④ User identifier</span>
          <span class="log-val">${userId || '<span style="color:#444">— not provided</span>'}</span>
        </div>
        <div class="log-field">
          <span class="log-key">⑤ Image metadata</span>
          <span class="log-val">${data.image_width} × ${data.image_height} px &nbsp;·&nbsp; ${selectedFile.type || 'unknown'}</span>
        </div>
        <div class="log-field">
          <span class="log-key">⑥ Faces detected</span>
          <span class="log-val">${data.face_count}</span>
        </div>
        <div class="log-field">
          <span class="log-key">⑦ Processing time</span>
          <span class="log-val">${data.processing_time_ms} ms</span>
        </div>
      </div>
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e1e1e;
                  font-size:.73rem;color:#555;display:flex;gap:16px;flex-wrap:wrap">
        <span>Saved to <code style="color:#888">logs/requests.db</code></span>
        <span>Written to <code style="color:#888">logs/app.log</code> (JSON)</span>
        <span>Thread-safe &middot; Unique request tracking</span>
        <a href="/ui/logs" style="color:#4ade80;margin-left:auto">View all logs →</a>
      </div>`;

    document.getElementById('results').style.display = 'block';
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    document.getElementById('spinner').style.display = 'none';
    detectBtn.disabled = false;
  }
}

function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = 'Error: ' + msg;
  box.style.display = 'block';
  document.getElementById('spinner').style.display = 'none';
  detectBtn.disabled = false;
}
</script>
</body>
</html>
"""



# ---------------------------------------------------------------------------
# Logs API + viewer
# ---------------------------------------------------------------------------

@ui_router.get("/api/logs", include_in_schema=False)
async def get_logs(
    limit: int = Query(50, le=500),
    detector: str = Query("all"),
    status: str = Query("all"),
):
    """Return recent request logs from SQLite as JSON."""
    db_path = LOG_DIR / "requests.db"
    if not db_path.exists():
        return JSONResponse({"rows": [], "total": 0})

    clauses = []
    if detector != "all":
        clauses.append(f"detector = '{detector}'")
    if status != "all":
        clauses.append(f"status = '{status}'")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM request_logs {where}").fetchone()[0]
        rows = conn.execute(
            f"""SELECT id, request_id, user_id, timestamp, detector,
                       image_width, image_height, image_format,
                       face_count, processing_time_ms, status, error_message
                FROM request_logs {where}
                ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return JSONResponse({"rows": [dict(r) for r in rows], "total": total})


@ui_router.get("/ui/logs", response_class=HTMLResponse, include_in_schema=False)
async def logs_ui():
    return HTMLResponse(content=_LOGS_HTML)


_LOGS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Logs — Face Detection Service</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#0f0f0f; color:#e0e0e0; min-height:100vh; padding:28px 20px; }
h1  { text-align:center; font-size:1.5rem; color:#fff; margin-bottom:4px; }
.sub { text-align:center; color:#666; font-size:.82rem; margin-bottom:20px; }

.nav { display:flex;justify-content:center;gap:20px;margin-bottom:22px; }
.nav a { text-decoration:none;font-size:.85rem; }
.nav a.active { color:#4ade80; }
.nav a:not(.active) { color:#888; }
.nav a:hover { color:#fff; }

/* ── Requirement checklist ── */
.checklist { max-width:900px;margin:0 auto 22px;background:#1a1a1a;
             border:1px solid #2a2a2a;border-radius:10px;padding:16px 20px; }
.checklist h3 { font-size:.78rem;color:#888;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:12px; }
.checklist-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px; }
.chk { background:#111;border:1px solid #222;border-radius:6px;padding:8px 12px;
        font-size:.78rem;display:flex;align-items:center;gap:8px; }
.chk .icon { color:#4ade80;font-size:1rem; }
.chk .text { color:#ccc; }

/* ── Toolbar ── */
.toolbar { display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;
           max-width:1100px;margin:0 auto 14px; }
.toolbar label { font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.04em; }
select,.btn-sm { padding:7px 13px;border-radius:7px;border:1px solid #333;font-size:.85rem;cursor:pointer; }
select { background:#1a1a1a;color:#e0e0e0; }
.btn-sm { background:#4ade80;color:#000;font-weight:700;border-color:#4ade80; }
.btn-sm:hover { background:#22c55e; }
.ts-info { margin-left:auto;font-size:.72rem;color:#555;align-self:flex-end;padding-bottom:2px; }

/* ── Stats ── */
.stats { display:flex;gap:10px;flex-wrap:wrap;max-width:1100px;margin:0 auto 18px; }
.stat  { background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;
          padding:10px 16px;flex:1;min-width:110px; }
.stat .val { font-size:1.3rem;font-weight:700;color:#4ade80; }
.stat .lbl { font-size:.68rem;color:#666;text-transform:uppercase; }

/* ── Table ── */
.wrap { max-width:1100px;margin:0 auto;overflow-x:auto; }
table { width:100%;border-collapse:collapse;font-size:.8rem; }
th { background:#1a1a1a;color:#777;padding:9px 11px;text-align:left;
     font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;
     border-bottom:1px solid #2a2a2a;white-space:nowrap;cursor:pointer; }
th:hover { color:#ccc; }
td { padding:8px 11px;border-bottom:1px solid #171717;vertical-align:top; }
tr { cursor:pointer; }
tr:hover td { background:#131313; }
tr.expanded td { background:#141414; }

.badge { display:inline-block;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:700; }
.ok    { background:#0d2d1a;color:#4ade80; }
.err   { background:#2d0d0d;color:#f87171; }
.haar  { background:#1a1a2d;color:#818cf8; }
.mediapipe { background:#1a2a1a;color:#4ade80; }
.yolo  { background:#2d1a0d;color:#fb923c; }
.benchmark { background:#2a2a0d;color:#facc15; }

.mono { font-family:monospace;font-size:.72rem;color:#888; }
.ts   { white-space:nowrap;color:#666;font-size:.73rem; }

/* ── Detail row ── */
.detail-row td { padding:0!important; border-bottom:1px solid #1a1a1a; }
.detail-box { padding:14px 16px;background:#0d0d0d;display:none; }
.detail-box.open { display:block; }
.fields { display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin-bottom:12px; }
.field { background:#111;border:1px solid #1e1e1e;border-radius:6px;padding:9px 12px; }
.fkey { display:block;font-size:.65rem;color:#555;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px; }
.fval { color:#ccc;font-size:.8rem;word-break:break-all; }
.fval.green { color:#4ade80; }
.fval.mono  { font-family:monospace;color:#60a5fa;font-size:.72rem; }

.spinner { text-align:center;padding:30px;color:#666; }
.empty   { text-align:center;padding:40px;color:#444; }
</style>
</head>
<body>

<h1>Request Logs</h1>
<p class="sub">Inspect, filter, and export detection requests</p>

<div class="nav">
  <a href="/ui">Detection</a>
  <a href="/ui/live">Live Camera</a>
  <a href="/ui/logs" class="active">Logs</a>
  <a href="/docs">API Docs</a>
</div>

<div class="toolbar">
  <div><label>Detector</label><br>
    <select id="detFilter" onchange="load()">
      <option value="all">All</option>
      <option value="haar">Haar</option>
      <option value="mediapipe">MediaPipe</option>
      <option value="yolo">YOLO</option>
      <option value="benchmark">Benchmark</option>
    </select>
  </div>
  <div><label>Status</label><br>
    <select id="statusFilter" onchange="load()">
      <option value="all">All</option>
      <option value="success">Success</option>
      <option value="error">Error</option>
    </select>
  </div>
  <div><label>Show last</label><br>
    <select id="limitSel" onchange="load()">
      <option value="20">20 rows</option>
      <option value="50" selected>50 rows</option>
      <option value="100">100 rows</option>
      <option value="500">500 rows</option>
    </select>
  </div>
  <button class="btn-sm" onclick="load()" style="margin-bottom:2px">↻ Refresh</button>
  <button class="btn-sm" onclick="exportLogs('csv')" style="margin-bottom:2px;background:#1a1a1a;color:#4ade80;border-color:#333">⬇ CSV</button>
  <button class="btn-sm" onclick="exportLogs('json')" style="margin-bottom:2px;background:#1a1a1a;color:#4ade80;border-color:#333">⬇ JSON</button>
  <span class="ts-info" id="tsInfo"></span>
</div>

<div class="stats">
  <div class="stat"><div class="val" id="sTotal">—</div><div class="lbl">Total requests</div></div>
  <div class="stat"><div class="val" id="sSuccess">—</div><div class="lbl">Successful</div></div>
  <div class="stat"><div class="val" id="sErrors">—</div><div class="lbl">Errors</div></div>
  <div class="stat"><div class="val" id="sAvgTime">—</div><div class="lbl">Avg time (ms)</div></div>
  <div class="stat"><div class="val" id="sAvgFaces">—</div><div class="lbl">Avg faces</div></div>
</div>

<div class="wrap">
  <div class="spinner" id="spinner">Loading…</div>
  <div class="empty"  id="empty"   style="display:none">No requests logged yet.</div>
  <table id="tbl" style="display:none">
    <thead>
      <tr>
        <th>#</th>
        <th>Timestamp</th>
        <th>Request ID</th>
        <th>Detector</th>
        <th>User</th>
        <th>Image</th>
        <th>Faces</th>
        <th>Time (ms)</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<script>
let allRows = [];

function detBadge(d) {
  return `<span class="badge ${d}">${d}</span>`;
}
function stBadge(s, err) {
  return s === 'success'
    ? `<span class="badge ok">ok</span>`
    : `<span class="badge err" title="${err||''}">error</span>`;
}

function exportLogs(format) {
  if (!allRows.length) { alert('Nothing to export — load some logs first.'); return; }
  const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
  let blob, filename;

  if (format === 'json') {
    blob = new Blob([JSON.stringify(allRows, null, 2)], { type: 'application/json' });
    filename = `request-logs-${ts}.json`;
  } else {
    const cols = ['id','timestamp','request_id','user_id','detector','image_width',
                  'image_height','image_format','face_count','processing_time_ms',
                  'status','error_message'];
    const esc = v => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      return /[",\\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
    };
    const lines = [cols.join(','), ...allRows.map(r => cols.map(c => esc(r[c])).join(','))];
    blob = new Blob([lines.join('\\n')], { type: 'text/csv' });
    filename = `request-logs-${ts}.csv`;
  }

  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function toggleDetail(idx) {
  const box = document.getElementById('detail-' + idx);
  if (!box) return;
  box.classList.toggle('open');
}

function buildDetailHTML(r) {
  const fmt = r.image_format || 'unknown';
  const size = `${r.image_width||'?'} × ${r.image_height||'?'} px`;
  return `
  <div class="fields">
    <div class="field">
      <span class="fkey">Timestamp</span>
      <span class="fval">${(r.timestamp||'').replace('T',' ').slice(0,23)} UTC</span>
    </div>
    <div class="field">
      <span class="fkey">Request ID</span>
      <span class="fval mono">${r.request_id||'—'}</span>
    </div>
    <div class="field">
      <span class="fkey">Detector</span>
      <span class="fval green">${r.detector}</span>
    </div>
    <div class="field">
      <span class="fkey">User</span>
      <span class="fval">${r.user_id || '<span style="color:#444">—</span>'}</span>
    </div>
    <div class="field">
      <span class="fkey">Image size</span>
      <span class="fval">${size}</span>
    </div>
    <div class="field">
      <span class="fkey">Image format</span>
      <span class="fval">${fmt}</span>
    </div>
    <div class="field">
      <span class="fkey">Faces detected</span>
      <span class="fval green">${r.face_count}</span>
    </div>
    <div class="field">
      <span class="fkey">Processing time</span>
      <span class="fval">${r.processing_time_ms?.toFixed(2)} ms</span>
    </div>
    ${r.error_message ? `<div class="field" style="grid-column:1/-1">
      <span class="fkey">Error</span>
      <span class="fval" style="color:#f87171">${r.error_message}</span>
    </div>` : ''}
  </div>
  <div style="font-size:.72rem;color:#555">
    Stored in <code>logs/requests.db</code> (SQLite) and <code>logs/app.log</code> (JSON lines)
  </div>`;
}

async function load() {
  const det    = document.getElementById('detFilter').value;
  const status = document.getElementById('statusFilter').value;
  const limit  = document.getElementById('limitSel').value;

  document.getElementById('spinner').style.display = 'block';
  document.getElementById('tbl').style.display     = 'none';
  document.getElementById('empty').style.display   = 'none';

  const resp = await fetch(`/api/logs?limit=${limit}&detector=${det}&status=${status}`);
  const data = await resp.json();
  allRows = data.rows;

  // Stats
  const ok  = allRows.filter(r => r.status === 'success').length;
  const err = allRows.filter(r => r.status === 'error').length;
  document.getElementById('sTotal').textContent   = data.total;
  document.getElementById('sSuccess').textContent = ok;
  document.getElementById('sErrors').textContent  = err;
  document.getElementById('sAvgTime').textContent = allRows.length
    ? (allRows.reduce((s,r) => s + r.processing_time_ms, 0) / allRows.length).toFixed(1) : '—';
  document.getElementById('sAvgFaces').textContent = allRows.length
    ? (allRows.reduce((s,r) => s + r.face_count, 0) / allRows.length).toFixed(1) : '—';

  if (allRows.length === 0) {
    document.getElementById('empty').style.display = 'block';
    document.getElementById('spinner').style.display = 'none';
    return;
  }

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  allRows.forEach((r, idx) => {
    const ts  = (r.timestamp||'').replace('T',' ').slice(0,19);
    const fmt = r.image_format || '—';
    const size= `${r.image_width||'?'}×${r.image_height||'?'}`;
    tbody.innerHTML += `
      <tr onclick="toggleDetail(${idx})">
        <td class="mono">${r.id}</td>
        <td class="ts">${ts}</td>
        <td class="mono" style="color:#60a5fa">${(r.request_id||'').slice(0,8)}…</td>
        <td>${detBadge(r.detector)}</td>
        <td class="mono">${r.user_id || '<span style="color:#333">—</span>'}</td>
        <td style="color:#888;font-size:.75rem">${size} &nbsp;·&nbsp; ${fmt}</td>
        <td style="font-weight:700;color:#fff">${r.face_count}</td>
        <td style="color:#888">${r.processing_time_ms?.toFixed(1)}</td>
        <td>${stBadge(r.status, r.error_message)}</td>
      </tr>
      <tr class="detail-row">
        <td colspan="9">
          <div class="detail-box" id="detail-${idx}">
            ${buildDetailHTML(r)}
          </div>
        </td>
      </tr>`;
  });

  document.getElementById('tbl').style.display    = 'table';
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('tsInfo').textContent   = 'Updated: ' + new Date().toLocaleTimeString();
}

load();
setInterval(load, 5000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Live camera detection
# ---------------------------------------------------------------------------

@ui_router.get("/ui/live", response_class=HTMLResponse, include_in_schema=False)
async def live_ui():
    return HTMLResponse(content=_LIVE_HTML)


_LIVE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Detection — Face Detection Service</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#0f0f0f; color:#e0e0e0; min-height:100vh; padding:28px 20px; }
h1  { text-align:center; font-size:1.5rem; color:#fff; margin-bottom:4px; }
.sub { text-align:center; color:#666; font-size:.82rem; margin-bottom:20px; }

.nav { display:flex;justify-content:center;gap:20px;margin-bottom:22px; }
.nav a { text-decoration:none;font-size:.85rem; }
.nav a.active { color:#4ade80; }
.nav a:not(.active) { color:#888; }
.nav a:hover { color:#fff; }

.card { background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;
        padding:24px;max-width:960px;margin:0 auto 24px; }

.toolbar { display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:18px; }

label { display:block;font-size:.78rem;color:#888;margin-bottom:6px;
        text-transform:uppercase;letter-spacing:.04em; }

select, .btn {
  padding:10px 18px;border-radius:8px;border:1px solid #333;
  font-size:.9rem;cursor:pointer;
}
select { background:#111;color:#e0e0e0; }
.btn { background:#4ade80;color:#000;font-weight:700;border-color:#4ade80;transition:background .15s; }
.btn:hover { background:#22c55e; }
.btn.stop { background:#ef4444;border-color:#ef4444; }
.btn.stop:hover { background:#dc2626; }
.btn:disabled { background:#2a2a2a;color:#555;border-color:#2a2a2a;cursor:not-allowed; }

.video-wrap { position:relative;max-width:720px;margin:0 auto;border-radius:10px;overflow:hidden;
              background:#000;aspect-ratio:4/3; }
.video-wrap video { width:100%;height:100%;object-fit:contain;display:block; }
.video-wrap canvas { position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none; }

.stats { display:flex;gap:12px;flex-wrap:wrap;margin-top:18px; }
.stat  { background:#111;border:1px solid #2a2a2a;border-radius:8px;
         padding:10px 16px;flex:1;min-width:100px; }
.stat .val { font-size:1.3rem;font-weight:700;color:#4ade80; }
.stat .lbl { font-size:.68rem;color:#666;text-transform:uppercase; }

.msg { text-align:center;padding:60px 20px;color:#666;font-size:.9rem; }
.error { background:#2a1010;border:1px solid #ff4444;border-radius:8px;
         padding:12px 16px;color:#ff8888;font-size:.85rem;margin-top:12px;display:none; }
</style>
</head>
<body>

<h1>Live Face Detection</h1>
<p class="sub">Real-time face tracking using your camera</p>

<div class="nav">
  <a href="/ui">Detection</a>
  <a href="/ui/live" class="active">Live Camera</a>
  <a href="/ui/logs">Logs</a>
  <a href="/docs">API Docs</a>
</div>

<div class="card">
  <div class="toolbar">
    <div>
      <label>Detector</label>
      <select id="detector">
        <option value="haar">Haar Cascade</option>
        <option value="mediapipe" selected>MediaPipe</option>
        <option value="yolo">YOLOv8-face</option>
      </select>
    </div>

    <div>
      <label>Min confidence: <span id="confVal">30%</span></label>
      <input type="range" id="minConf" min="0" max="1" step="0.05" value="0.3"
             style="width:120px;accent-color:#4ade80;vertical-align:middle"
             oninput="document.getElementById('confVal').textContent=Math.round(this.value*100)+'%'">
    </div>

    <div>
      <label>Capture interval</label>
      <select id="interval">
        <option value="200">Fast (~5 fps)</option>
        <option value="500" selected>Normal (~2 fps)</option>
        <option value="1000">Slow (~1 fps)</option>
      </select>
    </div>

    <div>
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" id="logRequests" style="accent-color:#4ade80">
        Save to log
      </label>
    </div>

    <button class="btn" id="startBtn" onclick="toggleCamera()">Start Camera</button>
  </div>

  <div class="error" id="errorBox"></div>

  <div id="videoContainer" style="display:none">
    <div class="video-wrap">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
    </div>

    <div class="stats">
      <div class="stat"><div class="val" id="sFaces">0</div><div class="lbl">Faces</div></div>
      <div class="stat"><div class="val" id="sTime">--</div><div class="lbl">Inference (ms)</div></div>
      <div class="stat"><div class="val" id="sFPS">--</div><div class="lbl">Detection FPS</div></div>
      <div class="stat"><div class="val" id="sDetector">--</div><div class="lbl">Detector</div></div>
      <div class="stat"><div class="val" id="sTotal">0</div><div class="lbl">Total detections</div></div>
    </div>
  </div>

  <div class="msg" id="placeholder">
    Click "Start Camera" to begin live face detection.<br>
    <span style="font-size:.78rem;color:#444;margin-top:8px;display:inline-block">
      Your camera feed stays in the browser. Frames are sent to the local server for processing.
    </span>
  </div>
</div>

<script>
const video     = document.getElementById('video');
const overlay   = document.getElementById('overlay');
const ctx       = overlay.getContext('2d');
const startBtn  = document.getElementById('startBtn');
const errorBox  = document.getElementById('errorBox');

let stream      = null;
let running     = false;
let loopTimer   = null;
let totalDets   = 0;
let lastFPS     = 0;
let fpsFrames   = 0;
let fpsTimer    = null;

// ── Capture a frame from the video as a base64 JPEG ──────────────
function captureFrame() {
  const c = document.createElement('canvas');
  c.width  = video.videoWidth;
  c.height = video.videoHeight;
  if (c.width === 0 || c.height === 0) return null;
  c.getContext('2d').drawImage(video, 0, 0);
  return c.toDataURL('image/jpeg', 0.8).split(',')[1]; // strip data URI prefix
}

// ── Draw bounding boxes + confidence on the overlay canvas ───────
function drawDetections(faces, imgW, imgH) {
  overlay.width  = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const scaleX = overlay.width  / imgW;
  const scaleY = overlay.height / imgH;

  faces.forEach(f => {
    const b = f.bbox;
    const x = b.x1 * scaleX;
    const y = b.y1 * scaleY;
    const w = (b.x2 - b.x1) * scaleX;
    const h = (b.y2 - b.y1) * scaleY;

    // Bounding box
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Confidence label
    const conf = (f.confidence * 100).toFixed(0) + '%';
    ctx.font = 'bold 14px sans-serif';
    const tw = ctx.measureText(conf).width;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(x, y - 20, tw + 10, 20);
    ctx.fillStyle = '#4ade80';
    ctx.fillText(conf, x + 5, y - 5);

    // Landmarks
    if (f.landmarks) {
      ctx.fillStyle = '#f87171';
      f.landmarks.forEach(lm => {
        ctx.beginPath();
        ctx.arc(lm.x * scaleX, lm.y * scaleY, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  });
}

// ── Detection loop ───────────────────────────────────────────────
async function detectLoop() {
  if (!running) return;

  const b64 = captureFrame();
  if (!b64) {
    loopTimer = setTimeout(detectLoop, 100);
    return;
  }

  const detector = document.getElementById('detector').value;
  const minConf  = document.getElementById('minConf').value;

  try {
    const resp = await fetch('/detect/base64', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: b64,
        detector: detector,
        min_confidence: parseFloat(minConf),
        return_annotated: false,
        log_request: document.getElementById('logRequests').checked,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      showError(err.detail?.message || 'Detection failed');
    } else {
      const data = await resp.json();
      drawDetections(data.faces, data.image_width, data.image_height);

      // Update stats
      document.getElementById('sFaces').textContent    = data.face_count;
      document.getElementById('sTime').textContent     = data.processing_time_ms.toFixed(0);
      document.getElementById('sDetector').textContent = data.detector;
      totalDets++;
      fpsFrames++;
      document.getElementById('sTotal').textContent = totalDets;
      errorBox.style.display = 'none';
    }
  } catch (e) {
    showError('Network error: ' + e.message);
  }

  if (running) {
    const interval = parseInt(document.getElementById('interval').value);
    loopTimer = setTimeout(detectLoop, interval);
  }
}

// ── FPS counter ──────────────────────────────────────────────────
function startFPSCounter() {
  fpsFrames = 0;
  fpsTimer = setInterval(() => {
    document.getElementById('sFPS').textContent = fpsFrames.toFixed(1);
    fpsFrames = 0;
  }, 1000);
}

// ── Start / Stop ─────────────────────────────────────────────────
async function toggleCamera() {
  if (running) {
    stopCamera();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    showError('Camera access denied or unavailable. Please allow camera permissions.');
    return;
  }

  running = true;
  totalDets = 0;
  document.getElementById('placeholder').style.display    = 'none';
  document.getElementById('videoContainer').style.display  = 'block';
  startBtn.textContent = 'Stop Camera';
  startBtn.classList.add('stop');
  errorBox.style.display = 'none';

  startFPSCounter();
  detectLoop();
}

function stopCamera() {
  running = false;
  if (loopTimer) clearTimeout(loopTimer);
  if (fpsTimer) clearInterval(fpsTimer);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  document.getElementById('videoContainer').style.display = 'none';
  document.getElementById('placeholder').style.display    = 'block';
  startBtn.textContent = 'Start Camera';
  startBtn.classList.remove('stop');
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = 'block';
}
</script>
</body>
</html>
"""
