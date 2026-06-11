import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import "./App.css";

const API = "http://localhost:8001/api";
const CLASSES = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"];

// ─── DEMO SAMPLE IMAGES (use real file paths in production) ─────────────────
const DEMO_SAMPLES = [
  { label: "RBC · MICROSCOPY", file: "rbc_microscopy.jpg", icon: "🔬" },
  { label: "LEUKOCYTES", file: "leukocytes.jpg", icon: "🧫" },
  { label: "STAINED SMEAR", file: "stained_smear.jpg", icon: "🩸" },
];

export default function App() {
  const [page, setPage] = useState("home"); // home | lab | about
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState(0);
  const [recentDetections, setRecentDetections] = useState([]);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchStats();
    fetchDetections();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API}/stats`);
      setStats(res.data);
      setRuns(res.data.total_runs);
    } catch (e) {}
  };

  const fetchDetections = async () => {
    try {
      const res = await axios.get(`${API}/detections?limit=10`);
      setRecentDetections(res.data.detections);
    } catch (e) {}
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
    }
  };

  const runDetection = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(`${API}/detect`, formData);
      setResult(res.data);
      fetchStats();
      fetchDetections();
    } catch (e) {
      alert("Detection failed. Is the backend running?");
    }
    setLoading(false);
  };

  const clearDetections = async () => {
    await axios.delete(`${API}/detections`);
    fetchDetections();
    fetchStats();
  };

  const deleteDetection = async (id) => {
    await axios.delete(`${API}/detections/${id}`);
    fetchDetections();
    fetchStats();
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="app">
      {/* ── NAVBAR ── */}
      <nav className="navbar">
        <div className="nav-left">
          <div className="nav-logo">
            <span className="logo-icon">🧬</span>
            <div>
              <div className="nav-subtitle">HAEMATOLOGY · CV LAB</div>
              <div className="nav-title">BGNet · Blood Group Detector</div>
            </div>
          </div>
        </div>
        <div className="nav-center">
          <span className="status-dot online"></span>
          <span className="status-text">MODEL ONLINE</span>
          <span className="divider">|</span>
          <span className="model-name">📈 BGNet-ResNet18-v1.0.0</span>
          <span className="divider">|</span>
          <span className="runs-badge">📋 RUNS: {String(runs).padStart(3, "0")}</span>
        </div>
        <div className="nav-right">
          <button className="nav-btn" onClick={() => setPage("about")}>ℹ ABOUT</button>
        </div>
      </nav>

      {/* ── HOME PAGE ── */}
      {page === "home" && (
        <div className="home">
          {/* Hero */}
          <div className="hero">
            <div className="hero-left">
              <div className="hero-badge">✦ B.TECH · SEMESTER 6 · MAJOR PROJECT</div>
              <h1 className="hero-title">
                Blood Group Detection using{" "}
                <span className="highlight">Image Processing</span> & Deep Learning
              </h1>
              <p className="hero-desc">
                An end-to-end pipeline that ingests a blood sample image, runs it through a
                classical computer-vision preprocessing chain (grayscale → Gaussian → Sobel →
                Otsu segmentation), and classifies it into one of the eight ABO/Rh groups using a
                convolutional neural network.
              </p>
              <div className="hero-actions">
                <button className="btn-primary" onClick={() => setPage("lab")}>
                  Open Detection Lab ↗
                </button>
                <div className="hero-meta">
                  <span>224×224</span> · <span>RGB</span> · <span>8 CLASSES</span> ·{" "}
                  <span>SOFTMAX</span>
                </div>
              </div>
            </div>
            <div className="hero-right">
              <div className="hero-bg-img">
                <img src="/rbc_microscopy.jpg" alt="blood smear" onError={(e) => (e.target.style.display = "none")} />
              </div>
            </div>
          </div>

          {/* Detection Lab Preview */}
          <DetectionLab
            preview={preview}
            file={file}
            result={result}
            loading={loading}
            handleFileChange={handleFileChange}
            runDetection={runDetection}
            resetForm={resetForm}
            recentDetections={recentDetections}
            clearDetections={clearDetections}
            deleteDetection={deleteDetection}
            stats={stats}
          />

          {/* Architecture Section */}
          <ArchitectureSection />
        </div>
      )}

      {/* ── LAB PAGE (standalone) ── */}
      {page === "lab" && (
        <div className="lab-page">
          <DetectionLab
            preview={preview}
            file={file}
            result={result}
            loading={loading}
            handleFileChange={handleFileChange}
            runDetection={runDetection}
            resetForm={resetForm}
            recentDetections={recentDetections}
            clearDetections={clearDetections}
            deleteDetection={deleteDetection}
            stats={stats}
          />
        </div>
      )}

      {/* ── ABOUT PAGE ── */}
      {page === "about" && (
        <div className="about-page">
          <h2>About BGNet</h2>
          <p>B.Tech Semester VI Major Project – Image Processing & Deep Learning.</p>
          <p>Stack: FastAPI · React · MongoDB · PyTorch · OpenCV</p>
          <button className="btn-primary" onClick={() => setPage("home")}>← Back</button>
        </div>
      )}
    </div>
  );
}

// ─── DETECTION LAB COMPONENT ─────────────────────────────────────────────────
function DetectionLab({
  preview, file, result, loading,
  handleFileChange, runDetection, resetForm,
  recentDetections, clearDetections, deleteDetection, stats
}) {
  return (
    <div className="detection-lab">
      <div className="section-label">DETECTION LAB</div>
      <div className="section-title-row">
        <h2>Run Inference on a Sample</h2>
        <div className="pipeline-legend">
          <span className="legend-item"><span className="leg-line red"></span>pipeline</span>
          <span className="legend-item"><span className="leg-line dark"></span>cnn</span>
          <span className="legend-item"><span className="leg-line gray"></span>verdict</span>
        </div>
      </div>

      <div className="lab-grid">
        {/* ── Panel 01: Sample Input ── */}
        <div className="panel panel-input">
          <div className="panel-header">
            <span className="panel-dot red"></span>
            <span className="panel-num">01 · SAMPLE INPUT</span>
            {file && <span className="panel-meta">{file.name} · {(file.size / 1024).toFixed(0)}KB</span>}
          </div>

          <div className="input-area">
            {preview ? (
              <img src={preview} alt="preview" className="preview-img" />
            ) : (
              <label className="upload-zone">
                <input type="file" accept="image/*" onChange={handleFileChange} hidden />
                <div className="upload-icon">📁</div>
                <div>Drop image or click to upload</div>
                <div className="upload-sub">JPG, PNG · Blood smear microscopy</div>
              </label>
            )}
          </div>

          <div className="input-actions">
            <button className="btn-run" onClick={runDetection} disabled={!file || loading}>
              {loading ? "Processing..." : "▶ Run Detection"}
            </button>
            <button className="btn-reset" onClick={resetForm}>↺</button>
          </div>

          {/* Demo Samples */}
          <div className="demo-label">DEMO SAMPLES</div>
          <div className="demo-samples">
            {DEMO_SAMPLES.map((s) => (
              <div key={s.label} className="demo-thumb">
                <div className="demo-icon">{s.icon}</div>
                <div className="demo-name">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Panel 03: Prediction ── */}
        <div className="panel panel-prediction">
          <div className="panel-header dark">
            <span className="panel-dot red"></span>
            <span className="panel-num">03 · PREDICTION</span>
            {result && <span className="panel-ok">✓ OK</span>}
          </div>

          {result ? (
            <PredictionResult result={result} />
          ) : (
            <div className="prediction-empty">
              <p>Upload a blood smear image and click Run Detection</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Panel 02: Preprocessing Pipeline ── */}
      {result && (
        <>
          <PipelinePanel pipeline={result.pipeline} />
          <RGBHistogramPanel rgb={result.rgb_histogram} />
          <FeaturePanel features={result.features} stats={stats} />
        </>
      )}

      {/* ── Panel 05: Recent Detections ── */}
      <div className="panel recent-panel">
        <div className="panel-header">
          <span>🕐 05 · RECENT DETECTIONS ({recentDetections.length})</span>
          <button className="btn-clear" onClick={clearDetections}>🗑 Clear</button>
        </div>
        <table className="detections-table">
          <thead>
            <tr>
              <th>SAMPLE</th><th>FILE</th><th>GROUP</th><th>CONF.</th><th>TIME</th><th></th>
            </tr>
          </thead>
          <tbody>
            {recentDetections.map((d) => (
              <tr key={d._id}>
                <td><div className="thumb-placeholder">?</div></td>
                <td>{d.filename || "—"}</td>
                <td><span className="blood-group-badge">{d.blood_group}</span></td>
                <td>{d.confidence}%</td>
                <td>{formatDate(d.timestamp)}</td>
                <td><button className="btn-del" onClick={() => deleteDetection(d._id)}>🗑</button></td>
              </tr>
            ))}
            {recentDetections.length === 0 && (
              <tr><td colSpan={6} style={{ textAlign: "center", padding: "20px", opacity: 0.5 }}>No detections yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── PREDICTION RESULT ───────────────────────────────────────────────────────
function PredictionResult({ result }) {
  const { prediction, inference_ms, batch } = result;
  return (
    <div className="prediction-content">
      <div className="pred-label">DETECTED BLOOD GROUP</div>
      <div className="pred-group">{prediction.blood_group}</div>
      <div className="pred-rh">
        <span className="rh-label">RH FACTOR</span>
        <span className="rh-value">{prediction.rh_factor}</span>
      </div>
      <div className="conf-row">
        <span className="conf-label">CONFIDENCE</span>
        <span className="conf-value">{prediction.confidence}%</span>
      </div>
      <div className="conf-bar">
        <div className="conf-fill" style={{ width: `${prediction.confidence}%` }}></div>
      </div>
      <div className="conf-meta">
        <span>softmax</span><span>argmax</span>
      </div>
      <div className="infer-meta">
        <span>⏱ {inference_ms}MS</span>
        <span>BATCH {batch}</span>
      </div>

      {/* Class Probabilities */}
      <div className="class-probs-label">CLASS PROBABILITIES</div>
      <div className="class-probs">
        {CLASSES.map((cls) => {
          const prob = prediction.class_probabilities[cls] || 0;
          const isTop = cls === prediction.blood_group;
          return (
            <div key={cls} className="prob-row">
              <span className={`prob-cls ${isTop ? "top" : ""}`}>{cls}</span>
              <div className="prob-bar-wrap">
                <div className={`prob-bar ${isTop ? "top" : ""}`} style={{ width: `${prob}%` }}></div>
              </div>
              <span className="prob-val">{prob}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── PIPELINE PANEL ──────────────────────────────────────────────────────────
function PipelinePanel({ pipeline }) {
  const stages = [
    { key: "acquisition", num: "01", label: "ACQUISITION", sub: "RGB · Raw", tech: "RGB Load", time: pipeline.timings.acquisition },
    { key: "grayscale", num: "02", label: "GRAYSCALE", sub: "Y = 0.299R + 0.587G + 0...", tech: "Luminance", time: pipeline.timings.grayscale },
    { key: "denoise", num: "03", label: "DENOISE", sub: "Gaussian · k=5 σ=1.2", tech: "Gaussian Blur", time: pipeline.timings.denoise },
    { key: "edge_detect", num: "04", label: "EDGE DETECT", sub: "Sobel · dx=1 dy=1", tech: "Sobel", time: pipeline.timings.edge_detect },
    { key: "segmentation", num: "05", label: "SEGMENTATION", sub: "Otsu Threshold", tech: "Otsu Threshold", time: pipeline.timings.segmentation },
  ];

  return (
    <div className="panel pipeline-panel">
      <div className="panel-header">
        <span className="panel-dot green"></span>
        <span>02 · PREPROCESSING PIPELINE</span>
        <span className="panel-meta right">{pipeline.image_size} · Otsu τ={pipeline.otsu_tau}</span>
      </div>
      <div className="pipeline-stages">
        {stages.map((s, i) => (
          <React.Fragment key={s.key}>
            <div className="stage">
              <div className="stage-img">
                {pipeline.images[s.key] ? (
                  <img src={`data:image/png;base64,${pipeline.images[s.key]}`} alt={s.label} />
                ) : (
                  <div className="stage-placeholder">—</div>
                )}
              </div>
              <div className="stage-info">
                <div className="stage-num">{s.num} · {s.label}</div>
                <div className="stage-sub">{s.sub}</div>
                <div className="stage-meta">
                  <span>{s.tech}</span>
                  <span className="stage-time">{s.time}ms</span>
                </div>
              </div>
            </div>
            {i < stages.length - 1 && <div className="stage-arrow">→</div>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// ─── RGB HISTOGRAM ───────────────────────────────────────────────────────────
function RGBHistogramPanel({ rgb }) {
  const data = rgb.bins.map((bin, i) => ({
    bin,
    R: rgb.channels.r[i],
    G: rgb.channels.g[i],
    B: rgb.channels.b[i],
  }));

  return (
    <div className="panel histogram-panel">
      <div className="panel-header">
        <span>RGB HISTOGRAM · 24 BINS</span>
        <span className="panel-meta right">channel distribution (%)</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} barSize={12}>
          <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="B" fill="#3b82f6" stackId="a" />
          <Bar dataKey="G" fill="#22c55e" stackId="a" />
          <Bar dataKey="R" fill="#ef4444" stackId="a" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── FEATURE PANEL ───────────────────────────────────────────────────────────
function FeaturePanel({ features, stats }) {
  const radarData = [
    { subject: "Intensity", value: (features.m_pixel / 255) * 100 },
    { subject: "Edges", value: features.edge_density * 100 },
    { subject: "Texture", value: (features.entropy / 8) * 100 },
    { subject: "Segment %", value: features.seg_area },
    { subject: "Red Ch.", value: (features.red_ch_mean / 255) * 100 },
    { subject: "Entropy", value: (features.entropy / 8) * 100 },
  ];

  const distData = stats?.distribution?.map((d) => ({
    name: d._id, count: d.count,
    fill: d._id?.includes("+") ? "#ef4444" : "#1f2937"
  })) || [];

  return (
    <div className="feature-row">
      <div className="panel feature-panel">
        <div className="panel-header">
          <span className="panel-dot green"></span>
          <span>04 · FEATURE EXTRACTION</span>
          <span className="panel-meta right">512-d vector</span>
        </div>
        <div className="feature-content">
          <div className="metrics-table">
            <div className="metric-label">METRIC DUMP</div>
            {[
              ["M(PIXEL)", `${features.m_pixel}/255`],
              ["Σ(PIXEL)", features.sigma_pixel],
              ["EDGE DENSITY", features.edge_density],
              ["EST. CELLS", features.est_cells],
              ["HOMOGENEITY", features.homogeneity],
              ["SEG. AREA", `${features.seg_area}%`],
              ["RED CH. M", features.red_ch_mean],
              ["RBC:WBC", features.rbc_wbc],
              ["ENTROPY", `${features.entropy}bits`],
            ].map(([k, v]) => (
              <div key={k} className="metric-row">
                <span className="metric-key">{k}</span>
                <span className="metric-val">{v}</span>
              </div>
            ))}
          </div>
          <div className="radar-wrap">
            <div className="metric-label">FEATURE RADAR</div>
            <RadarChart cx={140} cy={120} outerRadius={80} width={280} height={240} data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis tick={false} />
              <Radar dataKey="value" fill="#ef4444" fillOpacity={0.4} stroke="#ef4444" />
            </RadarChart>
          </div>
        </div>
      </div>

      {/* Distribution */}
      {stats && (
        <div className="panel dist-panel">
          <div className="panel-header">
            <span className="panel-dot green"></span>
            <span>DISTRIBUTION</span>
            <span className="panel-meta right">n = {stats.total_runs}</span>
          </div>
          <div className="dist-stats">
            <div className="dist-stat">
              <div className="dist-label">TOTAL RUNS</div>
              <div className="dist-val">{stats.total_runs}</div>
            </div>
            <div className="dist-stat">
              <div className="dist-label">AVG. CONF.</div>
              <div className="dist-val">{stats.avg_confidence}%</div>
            </div>
            <div className="dist-stat">
              <div className="dist-label">CLASSES</div>
              <div className="dist-val">{stats.classes}</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={distData}>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ─── ARCHITECTURE SECTION ────────────────────────────────────────────────────
function ArchitectureSection() {
  return (
    <div className="arch-section">
      <div className="section-label">TECHNICAL ABSTRACT</div>
      <div className="arch-title-row">
        <h2>Methodology & Architecture</h2>
        <span className="arch-version">v1.0.0 · Feb 2026</span>
      </div>
      <div className="arch-cards">
        {[
          {
            num: "01", icon: "🧪", title: "Dataset",
            desc: "Synthetic + public blood smear imagery resized to 224×224, 8-class balanced sampling.",
            meta: "1,240 train · 210 test"
          },
          {
            num: "02", icon: "⚙️", title: "Pipeline",
            desc: "OpenCV chain: Grayscale → Gaussian(5,1.2) → Sobel(3×3) → Otsu threshold → CNN input.",
            meta: "5 stages · ≈80ms"
          },
          {
            num: "03", icon: "🤖", title: "Model",
            desc: "BGNet — ResNet-18 backbone, 512-d feature head, 8-class softmax classifier.",
            meta: "~11.7M params · FP32"
          },
          {
            num: "04", icon: "📊", title: "Output",
            desc: "ABO group (A/B/AB/O) + Rh factor, per-class probabilities, feature vector summary.",
            meta: "JSON · REST /api/detect"
          },
        ].map((c) => (
          <div key={c.num} className="arch-card">
            <div className="arch-card-header">
              <span className="arch-icon">{c.icon}</span>
              <span className="arch-num">{c.num}</span>
            </div>
            <div className="arch-card-title">{c.title}</div>
            <div className="arch-card-desc">{c.desc}</div>
            <div className="arch-card-meta">{c.meta}</div>
          </div>
        ))}
      </div>

      {/* Pseudocode */}
      <div className="code-block">
        <div className="code-label">FORWARD PASS — PSEUDO-CODE</div>
        <pre>{`# blood_group_detector.py
img   = cv2.imread(path)               # HxWx3 BGR
gray  = cv2.cvtColor(img, COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (5,5), 1.2)
edges = cv2.Sobel(blur, cv2.CV_8U, 1, 1, ksize=3)
_, seg = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
x     = cv2.resize(img, (224,224)) / 255.0
logits = bgnet(x.transpose(2,0,1)[None, ...])
probs  = softmax(logits, axis=1)        # shape (1, 8)
pred   = CLASSES[int(probs.argmax())]   # e.g. "A+", "O-"`}</pre>
      </div>

      <div className="footer-meta">Major project · B.Tech Semester VI · Image Processing & Deep Learning.</div>
      <div className="footer">
        © 2026 · Blood Group Detection · Semester VI Major Project
        <span>FASTAPI · REACT · MONGODB · OPENCV</span>
      </div>
    </div>
  );
}

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.getDate().toString().padStart(2, "0")} ${d.toLocaleString("en", { month: "short" })} · ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")} ${d.getHours() >= 12 ? "PM" : "AM"}`;
}
