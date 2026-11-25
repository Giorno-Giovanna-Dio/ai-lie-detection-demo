// AI 微表情測謊分析 Demo
// Browser equivalent of the OpenCV + DeepFace pipeline (see python/analyze_video.py):
//   frames → MediaPipe Face Landmarker (landmarks + blendshapes ≈ FACS action units)
//          → face-api expression net (7 emotions, same classes as DeepFace)
//          → per-subject baseline (first CAL_SECONDS) → z-scores → weighted score 0–100.
// ?studio=1&video=…&start=…&end=… renders a 1920×1080 demo video (used to make media/*.mp4).

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACEAPI_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/dist/face-api.esm.js";
const FACEAPI_MODELS = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/";

const CAL_SECONDS = 10;
const LIME = "#d7ff3a", RED = "#ff4b3e", AMBER = "#ffb02e", BLUE = "#7aa7ff", INK = "#f2f1ec", MUTED = "#8b8d93";

// Cue definitions: weight in the final score, and a std floor so tiny baseline noise can't explode z.
const CUES = {
  brow:  { zh: "眉毛上揚",  en: "Brow raise",      w: 1.0, floor: 0.04 },
  gaze:  { zh: "眼神飄移",  en: "Gaze aversion",   w: 1.0, floor: 0.05 },
  blink: { zh: "眨眼頻率",  en: "Blink rate",      w: 0.6 },
  lip:   { zh: "抿嘴",      en: "Lip press",       w: 0.7, floor: 0.04 },
  head:  { zh: "頭部晃動",  en: "Head movement",   w: 0.5, floor: 4 },
  fear:  { zh: "驚訝/恐懼", en: "Fear / surprise", w: 0.8, floor: 0.06 },
  shift: { zh: "情緒突變",  en: "Emotion shift",   w: 0.5, floor: 0.06 },
  hr:    { zh: "心率上升",  en: "Heart rate",      w: 0.6 },
};
const KEYS = Object.keys(CUES);
const EMO_KEYS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];
const EMO_ZH = ["平靜", "開心", "難過", "生氣", "恐懼", "厭惡", "驚訝"];
const HOT = 1.8;     // z above which a cue is shown as active
const EVENT_Z = 2.2; // z that logs an event

// ---------- DOM ----------
const $ = (s) => document.querySelector(s);
const video = $("#src");
const canvas = $("#stage");
const ctx = canvas.getContext("2d");
const ppgCanvas = document.createElement("canvas");
ppgCanvas.width = 32; ppgCanvas.height = 16;
const ppgCtx = ppgCanvas.getContext("2d", { willReadFrequently: true });
let W = canvas.width, H = canvas.height;
const view = { x: 0, y: 0, w: W, h: H }; // where the video is drawn on the canvas

// ---------- state ----------
const S = {
  landmarker: null, faceapi: null,
  mode: null, mirrored: false, running: false, t0: 0,
  rec: null, camStream: null,
  studio: null, // studio render config/state
};
let A; // analysis state, reset on recalibration
function resetAnalysis() {
  A = {
    calStart: null, calibrated: false, samples: { brow: [], lip: [], head: [], fear: [], shift: [], gh: [], gv: [] },
    base: null, smooth: {}, zs: Object.fromEntries(KEYS.map((k) => [k, 0])), active: {},
    emo: null, emoBusy: false, lastEmo: -1, fear: 0, shift: 0,
    blinkClosed: false, blinks: [], gazeLast: null,
    headPrev: null, headSpeed: 0,
    score: 0, timeline: [], events: [],
    lastT: null, lastVideoTime: -1, face: false,
    ppg: { t: [], g: [], hr: null, conf: 0, last: 0, disabled: false, baseList: [] },
    lastUI: 0,
  };
  renderEvents();
}
resetAnalysis();

// ---------- helpers ----------
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ema = (prev, x, dt, tau) => (prev === undefined || prev === null ? x : prev + (x - prev) * (1 - Math.exp(-dt / tau)));
const meanStd = (arr) => {
  const n = arr.length || 1;
  const m = arr.reduce((s, x) => s + x, 0) / n;
  const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / n;
  return { mean: m, std: Math.sqrt(v) };
};
const median = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const fmtT = (t) => { const m = Math.floor(t / 60); const s = t - m * 60; return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`; };
const scoreColor = (s) => (s < 40 ? LIME : s < 70 ? AMBER : RED);
const clock = () => (S.mode === "file" ? video.currentTime : performance.now() / 1000 - S.t0);
const SANS = '"Noto Sans TC", "PingFang TC", system-ui, sans-serif';
const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
const DISPLAY = '"Space Grotesk", "Noto Sans TC", system-ui, sans-serif';

function setStatus(text, kind) {
  const st = $("#status"); if (!st) return;
  $("#statusText").textContent = text;
  st.className = `status ${kind}`;
}

// ---------- model loading ----------
async function loadModels() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MP_MODEL, delegate },
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO", numFaces: 1,
    });
    try { S.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
    catch { S.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
  } catch (e) {
    console.error(e);
    setStatus("臉部模型載入失敗，請檢查網路後重新整理", "error");
    return;
  }
  setStatus("臉部模型就緒 · 載入情緒模型中…", "loading");
  enableSources();
  try {
    const fa = await import(FACEAPI_URL);
    try { await fa.tf.setBackend("webgl"); } catch { /* default backend */ }
    await fa.tf.ready();
    await fa.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODELS);
    await fa.nets.faceExpressionNet.loadFromUri(FACEAPI_MODELS);
    S.faceapi = fa;
    setStatus("AI 模型就緒", "ready");
  } catch (e) {
    console.warn("face-api failed", e);
    setStatus("情緒模型載入失敗（其餘功能可用）", "error");
  }
}

function enableSources() {
  document.querySelectorAll('[data-action="cam"], input[data-action="file"]').forEach((el) => (el.disabled = false));
  document.querySelectorAll(".file-btn").forEach((el) => el.removeAttribute("aria-disabled"));
}

// ---------- sources ----------
function stopSource() {
  if (S.rec) S.rec.stop();
  if (S.camStream) { S.camStream.getTracks().forEach((t) => t.stop()); S.camStream = null; }
  video.pause();
  video.srcObject = null;
  video.removeAttribute("src");
  S.running = false;
}

async function startCam() {
  stopSource();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: true });
  } catch {
    try { stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
    catch (e) { alert("無法開啟鏡頭：" + e.message); return; }
  }
  S.camStream = stream;
  video.srcObject = stream;
  video.muted = true;
  S.mode = "cam"; S.mirrored = true;
  await video.play();
  begin();
}

function startFile(src, crossOrigin = false) {
  stopSource();
  if (crossOrigin) video.crossOrigin = "anonymous"; else video.removeAttribute("crossorigin");
  video.src = src;
  video.muted = false;
  S.mode = "file"; S.mirrored = false;
  video.play().catch(() => { video.muted = true; video.play(); });
  begin();
}

function begin() {
  S.t0 = performance.now() / 1000;
  resetAnalysis();
  S.running = true;
  $("#placeholder").classList.add("hidden");
  ["#playBtn", "#recalBtn", "#recBtn"].forEach((s) => ($(s).disabled = false));
  $("#playBtn").disabled = S.mode !== "file";
  $("#playBtn").textContent = "❚❚ 暫停";
}

video.addEventListener("ended", () => {
  if (S.studio) return;
  if (S.rec) S.rec.stop();
  const pb = $("#playBtn"); if (pb) pb.textContent = "▶ 重播";
});

// ---------- main loop ----------
function loop() {
  requestAnimationFrame(loop);
  if (S.studio) { studioTick(); return; }
  if (!S.running || !S.landmarker || video.readyState < 2) return;
  if (S.mode === "file" && video.currentTime === A.lastVideoTime) return; // paused / no new frame
  A.lastVideoTime = video.currentTime;
  try { processFrame(); } catch (e) { console.error(e); }
}
requestAnimationFrame(loop);

function fitCanvas() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;
  if (S.studio) {
    const box = S.studio.box;
    const s = Math.min(box.w / vw, box.h / vh);
    view.w = vw * s; view.h = vh * s;
    view.x = box.x + (box.w - view.w) / 2; view.y = box.y + (box.h - view.h) / 2;
    return;
  }
  const scale = Math.min(1, 1280 / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  W = w; H = h;
  view.x = 0; view.y = 0; view.w = W; view.h = H;
}

function drawVideo() {
  ctx.save();
  if (S.mirrored) { ctx.translate(view.x * 2 + view.w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, view.x, view.y, view.w, view.h);
  ctx.restore();
}

function processFrame(drawOverlay = true) {
  const t = clock();
  if (A.lastT !== null && t < A.lastT - 0.5) resetAnalysis(); // video was rewound
  const dt = A.lastT === null ? 1 / 30 : clamp(t - A.lastT, 1e-3, 0.25);
  A.lastT = t;

  const res = S.landmarker.detectForVideo(video, performance.now());
  fitCanvas();
  if (S.studio) { ctx.fillStyle = "#070708"; ctx.fillRect(0, 0, W, H); }
  drawVideo();

  const lm = res.faceLandmarks && res.faceLandmarks[0];
  A.face = !!lm;
  if (lm) {
    samplePPG(lm, t);
    const raw = extract(res, t, dt);
    maybeEmotion(t);
    updateAnalysis(raw, t, dt);
    drawFace(lm);
  } else {
    for (const k of KEYS) A.zs[k] = ema(A.zs[k], 0, dt, 0.8);
    A.score = ema(A.score, 0, dt, 1.5);
  }
  if (A.calibrated) A.timeline.push({ t, s: A.score });
  if (!S.studio && A.timeline.length > 3000) A.timeline.splice(0, A.timeline.length - 3000);
  if (drawOverlay) drawHUD(t);
  if (!S.studio && (t - A.lastUI > 0.1 || t < A.lastUI)) { A.lastUI = t; updatePanel(); }
}

// ---------- feature extraction ----------
function extract(res, t, dt) {
  const b = {};
  if (res.faceBlendshapes && res.faceBlendshapes[0]) for (const c of res.faceBlendshapes[0].categories) b[c.categoryName] = c.score;
  const g = (k) => b[k] || 0;

  const brow = Math.max(g("browInnerUp"), (g("browOuterUpLeft") + g("browOuterUpRight")) / 2);
  const blinkV = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2;
  if (!A.blinkClosed && blinkV > 0.5) { A.blinkClosed = true; A.blinks.push(t); }
  else if (A.blinkClosed && blinkV < 0.3) A.blinkClosed = false;
  A.blinks = A.blinks.filter((x) => t - x < 15 && x <= t);

  let gh = (g("eyeLookOutLeft") + g("eyeLookInRight") - (g("eyeLookInLeft") + g("eyeLookOutRight"))) / 2;
  let gv = (g("eyeLookUpLeft") + g("eyeLookUpRight") - (g("eyeLookDownLeft") + g("eyeLookDownRight"))) / 2;
  if (blinkV > 0.4 && A.gazeLast) [gh, gv] = A.gazeLast; // gaze is meaningless mid-blink
  A.gazeLast = [gh, gv];

  const lip = Math.max((g("mouthPressLeft") + g("mouthPressRight")) / 2, (g("mouthRollLower") + g("mouthRollUpper")) / 2);

  const M = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0] && res.facialTransformationMatrixes[0].data;
  if (M) {
    const yaw = Math.asin(clamp(-M[2], -1, 1)) * 57.2958;
    const pitch = Math.atan2(M[6], M[10]) * 57.2958;
    if (A.headPrev) A.headSpeed = ema(A.headSpeed, Math.hypot(yaw - A.headPrev[0], pitch - A.headPrev[1]) / dt, dt, 0.4);
    A.headPrev = [yaw, pitch];
  }
  return { brow, gh, gv, lip, head: A.headSpeed, fear: A.fear, shift: A.shift };
}

function maybeEmotion(t) {
  if (!S.faceapi || A.emoBusy || t - A.lastEmo < 0.25) return;
  A.emoBusy = true; A.lastEmo = t;
  const fa = S.faceapi;
  const token = A;
  fa.detectSingleFace(video, new fa.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 }))
    .withFaceExpressions()
    .then((r) => {
      if (!r || token !== A) return;
      const v = EMO_KEYS.map((k) => r.expressions[k] || 0);
      if (A.emo) A.shift = v.reduce((s, x, i) => s + Math.abs(x - A.emo[i]), 0) / 2;
      A.emo = v;
      A.fear = (r.expressions.fearful || 0) + 0.7 * (r.expressions.surprised || 0);
    })
    .catch(() => {})
    .finally(() => { token.emoBusy = false; });
}

// ---------- rPPG (camera heart rate) ----------
function samplePPG(lm, t) {
  const P = A.ppg;
  if (P.disabled) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const fw = Math.abs(lm[454].x - lm[234].x) * vw;
  const cx = lm[151].x * vw, cy = lm[151].y * vh; // forehead centre
  const w = Math.max(8, fw * 0.3), h = Math.max(6, fw * 0.12);
  let d;
  try {
    ppgCtx.drawImage(video, cx - w / 2, cy - h / 2, w, h, 0, 0, 32, 16);
    d = ppgCtx.getImageData(0, 0, 32, 16).data;
  } catch { P.disabled = true; return; }
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
  const sum = r + g + b;
  if (!sum) return;
  P.t.push(t); P.g.push(g / sum);
  while (P.t.length && t - P.t[0] > 12) { P.t.shift(); P.g.shift(); }
  if (t - P.last >= 1 || t < P.last) { P.last = t; estimateHR(); }
}

function estimateHR() {
  const P = A.ppg;
  if (P.t.length < 60) return;
  const span = P.t[P.t.length - 1] - P.t[0];
  if (span < 6) return;
  const fs = 30, N = Math.floor(span * fs), t0 = P.t[0];
  const x = new Float64Array(N);
  for (let i = 0, j = 0; i < N; i++) {
    const tt = t0 + i / fs;
    while (j < P.t.length - 2 && P.t[j + 1] < tt) j++;
    const a = P.t[j], b = P.t[j + 1];
    const f = b > a ? clamp((tt - a) / (b - a), 0, 1) : 0;
    x[i] = P.g[j] + (P.g[j + 1] - P.g[j]) * f;
  }
  const pre = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) pre[i + 1] = pre[i] + x[i];
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) { const lo = Math.max(0, i - 22), hi = Math.min(N - 1, i + 22); y[i] = x[i] - (pre[hi + 1] - pre[lo]) / (hi - lo + 1); }
  const { std } = meanStd(Array.from(y));
  if (!std) return;
  for (let i = 0; i < N; i++) y[i] = (y[i] / std) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
  let best = 0, bestF = 0, tot = 0, cnt = 0;
  for (let f = 0.75; f <= 3.0; f += 1 / 60) {
    const w = (2 * Math.PI * f) / fs;
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) { re += y[i] * Math.cos(w * i); im -= y[i] * Math.sin(w * i); }
    const p = re * re + im * im;
    tot += p; cnt++;
    if (p > best) { best = p; bestF = f; }
  }
  P.conf = best / (tot / cnt);
  if (P.conf > 2.5) {
    const bpm = bestF * 60;
    P.hr = P.hr ? P.hr * 0.6 + bpm * 0.4 : bpm;
    if (!A.calibrated) P.baseList.push(bpm);
  }
}

// ---------- baseline + scoring ----------
function updateAnalysis(raw, t, dt) {
  const sm = A.smooth;
  for (const k of ["brow", "lip", "fear", "shift"]) sm[k] = ema(sm[k], raw[k], dt, 0.25);
  sm.head = raw.head;
  sm.gh = ema(sm.gh, raw.gh, dt, 0.15);
  sm.gv = ema(sm.gv, raw.gv, dt, 0.15);

  if (A.calStart === null) A.calStart = t;
  if (!A.calibrated) {
    for (const k of Object.keys(A.samples)) A.samples[k].push(sm[k]);
    if (t - A.calStart >= CAL_SECONDS && A.samples.brow.length >= 30) finishCalibration(t);
    return;
  }

  const B = A.base;
  const z = {};
  for (const k of ["brow", "lip", "head", "fear", "shift"]) z[k] = (sm[k] - B[k].mean) / Math.max(B[k].std, CUES[k].floor);
  const gd = Math.hypot(sm.gh - B.gh, sm.gv - B.gv);
  z.gaze = (gd - B.gaze.mean) / Math.max(B.gaze.std, CUES.gaze.floor);
  const win = Math.min(15, Math.max(5, t - A.calStart));
  const rate = (A.blinks.length / win) * 60;
  z.blink = (rate - B.blinkRate) / Math.max(8, B.blinkRate * 0.6);
  if (!B.hr && A.ppg.hr) B.hr = A.ppg.hr;
  z.hr = A.ppg.hr && B.hr ? (A.ppg.hr - B.hr) / 6 : 0;

  let s = 0;
  for (const k of KEYS) {
    A.zs[k] = ema(A.zs[k], clamp(z[k] || 0, 0, 5), dt, 0.5);
    s += CUES[k].w * Math.max(0, A.zs[k] - 1);
    if (!A.active[k] && A.zs[k] > EVENT_Z) { A.active[k] = true; A.events.unshift({ t, k }); renderEvents(); }
    else if (A.active[k] && A.zs[k] < 1.4) A.active[k] = false;
  }
  const target = 100 / (1 + Math.exp(-(s - 1.8) * 1.3));
  A.score = ema(A.score, target, dt, 0.6);
}

function finishCalibration(t) {
  const sm = A.samples;
  const B = {};
  for (const k of ["brow", "lip", "head", "fear", "shift"]) B[k] = meanStd(sm[k]);
  B.gh = meanStd(sm.gh).mean; B.gv = meanStd(sm.gv).mean;
  B.gaze = meanStd(sm.gh.map((x, i) => Math.hypot(x - B.gh, sm.gv[i] - B.gv)));
  B.blinkRate = (A.blinks.filter((x) => x >= A.calStart).length / Math.max(1, t - A.calStart)) * 60;
  B.hr = median(A.ppg.baseList);
  A.base = B;
  A.calibrated = true;
  A.score = 100 / (1 + Math.exp(1.8 * 1.3));
}

// ---------- drawing ----------
const P = (p) => ({ x: view.x + (S.mirrored ? 1 - p.x : p.x) * view.w, y: view.y + p.y * view.h });
function conns(lm, list, color, lw) {
  ctx.beginPath();
  for (const { start, end } of list) { const a = P(lm[start]), b = P(lm[end]); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
}

function drawFace(lm) {
  const lw = Math.max(1, view.w / 700);
  const col = (k) => (A.zs[k] > HOT ? RED : LIME);
  const F = FaceLandmarker;
  conns(lm, F.FACE_LANDMARKS_TESSELATION, "rgba(242,241,236,0.09)", 0.6);
  conns(lm, F.FACE_LANDMARKS_FACE_OVAL, "rgba(215,255,58,0.45)", lw);
  conns(lm, F.FACE_LANDMARKS_LEFT_EYEBROW, col("brow"), lw * 2);
  conns(lm, F.FACE_LANDMARKS_RIGHT_EYEBROW, col("brow"), lw * 2);
  conns(lm, F.FACE_LANDMARKS_LEFT_EYE, col("blink"), lw);
  conns(lm, F.FACE_LANDMARKS_RIGHT_EYE, col("blink"), lw);
  conns(lm, F.FACE_LANDMARKS_LEFT_IRIS, col("gaze"), lw * 1.5);
  conns(lm, F.FACE_LANDMARKS_RIGHT_IRIS, col("gaze"), lw * 1.5);
  conns(lm, F.FACE_LANDMARKS_LIPS, col("lip"), lw * 1.5);

  // corner-bracket box around the face
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of lm) { const q = P(p); x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
  const pad = (x1 - x0) * 0.08;
  const bx = x0 - pad, by = y0 - pad, bw = x1 - x0 + 2 * pad, bh = y1 - y0 + 2 * pad;
  const c = Math.min(bw, bh) * 0.15;
  ctx.strokeStyle = A.calibrated ? scoreColor(A.score) : BLUE;
  ctx.lineWidth = lw * 2;
  ctx.beginPath();
  [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]].forEach(([x, y, sx, sy]) => {
    ctx.moveTo(x + sx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * c);
  });
  ctx.stroke();
  ctx.font = `600 ${Math.max(10, view.w / 70)}px ${MONO}`;
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fillText("SUBJECT 01", bx, by - 8);
}

function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }

function gauge(gx, gy, R, u, labelSize) {
  const a0 = Math.PI * 0.75, span = Math.PI * 1.5;
  ctx.lineCap = "round"; ctx.lineWidth = R * 0.14;
  ctx.strokeStyle = "rgba(242,241,236,0.1)";
  ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span); ctx.stroke();
  ctx.textAlign = "center";
  if (A.calibrated) {
    ctx.strokeStyle = scoreColor(A.score);
    ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span * (A.score / 100)); ctx.stroke();
    ctx.fillStyle = INK; ctx.font = `600 ${R * 0.6}px ${DISPLAY}`;
    ctx.fillText(`${Math.round(A.score)}%`, gx, gy + R * 0.18);
    ctx.fillStyle = scoreColor(A.score); ctx.font = `700 ${labelSize}px ${SANS}`;
    ctx.fillText("說謊可能性", gx, gy + R * 0.55);
  } else {
    const t = clock();
    const p = A.calStart === null ? 0 : clamp((t - A.calStart) / CAL_SECONDS, 0, 1);
    ctx.strokeStyle = BLUE;
    ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span * p); ctx.stroke();
    ctx.fillStyle = BLUE; ctx.font = `700 ${labelSize * 1.3}px ${SANS}`;
    ctx.fillText("建立基準線", gx, gy + R * 0.05);
    ctx.font = `400 ${labelSize}px ${MONO}`; ctx.fillStyle = INK;
    ctx.fillText(`${Math.ceil(CAL_SECONDS * (1 - p))}s`, gx, gy + R * 0.38);
  }
  ctx.textAlign = "left"; ctx.lineCap = "butt";
}

function drawHUD(t) {
  if (S.studio) return drawStudioPanel(t);
  const u = Math.min(W, H) / 720;
  roundRect(16 * u, 16 * u, 330 * u, 62 * u, 10 * u);
  ctx.fillStyle = "rgba(7,7,8,0.72)"; ctx.fill();
  ctx.fillStyle = LIME; ctx.font = `600 ${12 * u}px ${MONO}`;
  ctx.fillText("AI MICRO-EXPRESSION ANALYSIS", 30 * u, 40 * u);
  ctx.fillStyle = INK; ctx.font = `400 ${16 * u}px ${MONO}`;
  ctx.fillText(fmtT(Math.max(0, t)), 30 * u, 64 * u);
  if (S.rec) {
    const on = Math.floor(performance.now() / 500) % 2 === 0;
    ctx.fillStyle = on ? RED : "rgba(255,75,62,0.4)";
    ctx.beginPath(); ctx.arc(250 * u, 59 * u, 6 * u, 0, Math.PI * 2); ctx.fill();
    ctx.font = `600 ${13 * u}px ${MONO}`; ctx.fillText("REC", 262 * u, 64 * u);
  }

  const R = 78 * u, gx = W - R - 34 * u, gy = R + 34 * u;
  ctx.fillStyle = "rgba(7,7,8,0.72)";
  ctx.beginPath(); ctx.arc(gx, gy, R + 18 * u, 0, Math.PI * 2); ctx.fill();
  gauge(gx, gy, R, u, 13 * u);

  if (A.calibrated) {
    const hot = KEYS.filter((k) => A.zs[k] > HOT).sort((a, b) => A.zs[b] - A.zs[a]).slice(0, 4);
    let cy = gy + R + 34 * u;
    ctx.font = `500 ${14 * u}px ${SANS}`;
    for (const k of hot) {
      const label = `▲ ${CUES[k].zh}`;
      const tw = ctx.measureText(label).width + 20 * u;
      const x = W - 16 * u - tw;
      roundRect(x, cy, tw, 28 * u, 14 * u);
      ctx.fillStyle = "rgba(255,75,62,0.9)"; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(label, x + 10 * u, cy + 19 * u);
      cy += 34 * u;
    }
  }

  const th = 70 * u, tx = 16 * u, tw = W - 32 * u, ty = H - th - 16 * u;
  roundRect(tx, ty, tw, th, 10 * u);
  ctx.fillStyle = "rgba(7,7,8,0.72)"; ctx.fill();
  ctx.font = `400 ${10 * u}px ${MONO}`; ctx.fillStyle = MUTED;
  ctx.fillText("LIKELIHOOD · LAST 30s", tx + 10 * u, ty + 15 * u);
  const tl = A.timeline;
  if (tl.length > 1) {
    const tEnd = tl[tl.length - 1].t;
    drawTimeline(tx + 10 * u, ty + 22 * u, tw - 20 * u, th - 30 * u, tEnd - 30, tEnd, u, false);
  }

  if (!A.face) {
    ctx.textAlign = "center"; ctx.font = `700 ${20 * u}px ${SANS}`;
    ctx.fillStyle = AMBER;
    ctx.fillText("未偵測到臉部 · 請面向鏡頭", W / 2, H / 2);
    ctx.textAlign = "left";
  }
}

function drawTimeline(px0, py0, pw, ph, tStart, tEnd, u, labels, marks = []) {
  const span = Math.max(1e-3, tEnd - tStart);
  const xOf = (tt) => px0 + ((tt - tStart) / span) * pw;
  const yOf = (s) => py0 + ph - (s / 100) * ph;
  for (const m of marks) {
    if (m.kind !== "key") continue;
    ctx.fillStyle = "rgba(255,75,62,0.12)";
    ctx.fillRect(xOf(Math.max(m.t0, tStart)), py0, xOf(Math.min(m.t1, tEnd)) - xOf(Math.max(m.t0, tStart)), ph);
  }
  ctx.setLineDash([4 * u, 4 * u]); ctx.strokeStyle = "rgba(255,75,62,0.55)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px0, yOf(70)); ctx.lineTo(px0 + pw, yOf(70)); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath();
  let started = false;
  for (const p of A.timeline) { if (p.t < tStart) continue; const x = xOf(p.t), y = yOf(p.s); started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), (started = true)); }
  ctx.strokeStyle = scoreColor(A.score); ctx.lineWidth = 2 * u; ctx.stroke();
  for (const ev of A.events) {
    if (ev.t < tStart) continue;
    const x = xOf(ev.t);
    ctx.fillStyle = RED; ctx.beginPath(); ctx.arc(x, py0 + 4 * u, 3.5 * u, 0, Math.PI * 2); ctx.fill();
    if (labels) { ctx.save(); ctx.translate(x + 4 * u, py0 - 4 * u); ctx.rotate(-Math.PI / 4); ctx.font = `500 ${12 * u}px ${SANS}`; ctx.fillStyle = "rgba(255,75,62,0.9)"; ctx.fillText(CUES[ev.k].zh, 0, 0); ctx.restore(); }
  }
}

// ---------- side panel (page) ----------
const cueList = $("#cues");
if (cueList) cueList.innerHTML = KEYS.map((k) => `<li data-k="${k}"><span>${CUES[k].zh}</span><span class="bar"><i></i></span><span class="v">0.0</span></li>`).join("");
const emoList = $("#emos");
if (emoList) emoList.innerHTML = EMO_ZH.map((n) => `<li><span>${n}</span><span class="bar"><i></i></span><span class="v">0</span></li>`).join("");

function updatePanel() {
  const phase = $("#phase"); if (!phase) return;
  if (!S.running) { phase.textContent = "待機"; phase.className = "pill"; }
  else if (!A.calibrated) { phase.textContent = "校準中"; phase.className = "pill cal"; }
  else { phase.textContent = "分析中"; phase.className = "pill live"; }
  const sc = A.calibrated ? Math.round(A.score) : null;
  $("#scoreNum").textContent = sc === null ? "--" : sc;
  $("#scoreNum").style.color = sc === null ? "" : scoreColor(sc);
  $("#scoreFill").style.width = `${sc || 0}%`;
  $("#scoreFill").style.background = scoreColor(sc || 0);
  for (const li of cueList.children) {
    const z = A.zs[li.dataset.k] || 0;
    li.classList.toggle("hot", z > HOT);
    li.querySelector("i").style.width = `${clamp(z / 4, 0, 1) * 100}%`;
    li.querySelector(".v").textContent = z.toFixed(1);
  }
  const emo = A.emo || EMO_KEYS.map(() => 0);
  [...emoList.children].forEach((li, i) => {
    li.querySelector("i").style.width = `${emo[i] * 100}%`;
    li.querySelector(".v").textContent = Math.round(emo[i] * 100);
  });
  $("#hrNum").textContent = A.ppg.hr ? Math.round(A.ppg.hr) : "--";
  $("#hrNote").textContent = A.ppg.disabled ? "此影片來源無法讀取像素，心率停用" : A.base && A.base.hr ? `基準線 ${Math.round(A.base.hr)} bpm（實驗性）` : "由額頭膚色微小變化估計（實驗性）";
}

function renderEvents() {
  const ol = $("#events"); if (!ol) return;
  if (!A || !A.events.length) { ol.innerHTML = '<li class="empty">尚無事件</li>'; return; }
  ol.innerHTML = A.events.slice(0, 40).map((e) => `<li><span class="t">${fmtT(e.t)}</span><span>${CUES[e.k].zh}</span></li>`).join("");
}

// ---------- recording ----------
function makeRecorder(onBlob) {
  const stream = canvas.captureStream(30);
  try {
    if (S.mode === "cam" && S.camStream) S.camStream.getAudioTracks().forEach((tr) => stream.addTrack(tr));
    else if (S.mode === "file") {
      const vs = video.captureStream ? video.captureStream() : video.mozCaptureStream ? video.mozCaptureStream() : null;
      if (vs) vs.getAudioTracks().forEach((tr) => stream.addTrack(tr));
    }
  } catch { /* video only */ }
  const mime = ["video/mp4;codecs=avc1.42E01E,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm"].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : {});
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => onBlob(new Blob(chunks, { type: rec.mimeType || "video/webm" }));
  rec.start(1000);
  return rec;
}

function toggleRec() {
  if (S.rec) { S.rec.stop(); return; }
  try {
    S.rec = makeRecorder((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `ai-lie-detection-demo-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${blob.type.includes("mp4") ? "mp4" : "webm"}`;
      document.body.appendChild(a); a.click(); a.remove();
      S.rec = null;
      $("#recBtn").classList.remove("on");
      $("#recBtn").textContent = "● 錄製示範影片";
    });
  } catch (e) { alert("此瀏覽器不支援錄影：" + e.message); return; }
  $("#recBtn").classList.add("on");
  $("#recBtn").textContent = "■ 停止並下載";
}

// ---------- studio render (1920×1080 demo video) ----------
const INTRO = 3.5, OUTRO = 6;
async function runStudio(p) {
  document.documentElement.classList.add("studio");
  canvas.width = W = 1920; canvas.height = H = 1080;
  const st = S.studio = {
    ...p, box: { x: 48, y: 128, w: 1180, h: 704 }, phase: "load", t0: 0, recT: 0, summary: null,
  };
  window.__studio = { phase: "load" };
  video.crossOrigin = "anonymous";
  video.src = p.video;
  video.muted = false;
  await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
  video.currentTime = p.start;
  await new Promise((r) => video.addEventListener("seeked", r, { once: true }));
  S.mode = "file"; S.mirrored = false;
  resetAnalysis();
  fitCanvas();
  await document.fonts.ready;
  st.rec = makeRecorder(async (blob) => {
    window.__studio = { phase: "upload", bytes: blob.size };
    const r = await fetch(`/save?name=${encodeURIComponent(p.name)}.${blob.type.includes("mp4") ? "mp4" : "webm"}`, { method: "POST", body: blob });
    window.__studio = { phase: "done", bytes: blob.size, status: r.status, summary: st.summary };
  });
  st.phase = "intro"; st.recT = performance.now() / 1000;
  window.__studio = { phase: "intro" };
}

function studioTick() {
  const st = S.studio;
  if (st.phase === "load" || st.phase === "end") return;
  const now = performance.now() / 1000;
  if (st.phase === "intro") {
    drawIntro(now - st.recT);
    if (now - st.recT > INTRO) {
      st.phase = "play"; window.__studio = { phase: "play" };
      video.play().catch(() => { video.muted = true; video.play(); });
    }
    return;
  }
  if (st.phase === "play") {
    if (video.currentTime >= st.end || video.ended) {
      video.pause();
      st.summary = summarize();
      st.phase = "outro"; st.outroT = now; window.__studio = { phase: "outro" };
      return;
    }
    if (video.readyState < 2 || video.currentTime === A.lastVideoTime) return;
    A.lastVideoTime = video.currentTime;
    window.__studio = { phase: "play", t: video.currentTime };
    try { processFrame(); } catch (e) { console.error(e); }
    return;
  }
  if (st.phase === "outro") {
    drawOutro(now - st.outroT);
    if (now - st.outroT > OUTRO) { st.phase = "end"; st.rec.stop(); }
  }
}

// Offline render: steps through the clip frame by frame and encodes H.264 with WebCodecs,
// so output is a smooth 30 fps no matter how fast this machine (or a hidden tab) runs.
async function runStudioOffline(p) {
  document.documentElement.classList.add("studio");
  canvas.width = W = 1920; canvas.height = H = 1080;
  const st = S.studio = { ...p, box: { x: 48, y: 128, w: 1180, h: 704 }, phase: "offline" };
  window.__studio = { phase: "load" };
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.preload = "auto";
  const ready = new Promise((r) => video.addEventListener("loadeddata", r, { once: true }));
  video.src = p.video;
  if (video.readyState < 2) await ready;
  S.mode = "file"; S.mirrored = false;
  resetAnalysis();
  fitCanvas();
  await document.fonts.ready;

  const { Muxer, ArrayBufferTarget } = await import("https://cdn.jsdelivr.net/npm/mp4-muxer@5.1.3/build/mp4-muxer.mjs");
  const FPS = 30;
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: "avc", width: W, height: H }, fastStart: "in-memory" });
  const encoder = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: (e) => (window.__studio = { phase: "error", msg: String(e) }) });
  encoder.configure({ codec: "avc1.640028", width: W, height: H, bitrate: 5_000_000, framerate: FPS });
  let n = 0;
  const emit = async () => {
    const f = new VideoFrame(canvas, { timestamp: Math.round((n * 1e6) / FPS), duration: Math.round(1e6 / FPS) });
    encoder.encode(f, { keyFrame: n % 60 === 0 });
    f.close(); n++;
    while (encoder.encodeQueueSize > 8) await new Promise((r) => encoder.addEventListener("dequeue", r, { once: true }));
  };
  const seekTo = (t) => new Promise((r) => { video.addEventListener("seeked", r, { once: true }); video.currentTime = t; });

  for (let i = 0; i < INTRO * FPS; i++) { drawIntro(i / FPS); await emit(); }
  const frames = Math.round((p.end - p.start) * FPS);
  for (let i = 0; i < frames; i++) {
    const t = p.start + i / FPS;
    await seekTo(t);
    if (S.faceapi && t - A.lastEmo >= 0.25) {
      A.lastEmo = t;
      const fa = S.faceapi;
      let r = null;
      try { r = await fa.detectSingleFace(video, new fa.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })).withFaceExpressions(); } catch { /* skip */ }
      if (r) {
        const v = EMO_KEYS.map((k) => r.expressions[k] || 0);
        if (A.emo) A.shift = v.reduce((s, x, j) => s + Math.abs(x - A.emo[j]), 0) / 2;
        A.emo = v;
        A.fear = (r.expressions.fearful || 0) + 0.7 * (r.expressions.surprised || 0);
      }
    }
    processFrame();
    await emit();
    if (i % 15 === 0) window.__studio = { phase: "render", t, done: i, frames };
  }
  st.summary = summarize();
  for (let i = 0; i < OUTRO * FPS; i++) { drawOutro(i / FPS); await emit(); }
  await encoder.flush();
  muxer.finalize();
  const blob = new Blob([muxer.target.buffer], { type: "video/mp4" });
  window.__studio = { phase: "upload", bytes: blob.size };
  const res = await fetch(`/save?name=${encodeURIComponent(p.name)}.mp4`, { method: "POST", body: blob });
  window.__studio = { phase: "done", bytes: blob.size, status: res.status, summary: st.summary };
}

function markAt(t) { return (S.studio.marks || []).find((m) => t >= m.t0 && t < m.t1); }

function summarize() {
  const tl = A.timeline;
  const avg = (a) => (a.length ? a.reduce((s, p) => s + p.s, 0) / a.length : null);
  const segs = (S.studio.marks || []).map((m) => ({ ...m, avg: avg(tl.filter((p) => p.t >= m.t0 && p.t < m.t1)), peak: Math.max(0, ...tl.filter((p) => p.t >= m.t0 && p.t < m.t1).map((p) => p.s)) }));
  const counts = {};
  for (const e of A.events) counts[e.k] = (counts[e.k] || 0) + 1;
  return { overall: avg(tl), peak: Math.max(0, ...tl.map((p) => p.s)), segs, events: A.events.length, counts };
}

function topBar(t) {
  ctx.fillStyle = LIME; ctx.font = `600 18px ${MONO}`;
  ctx.fillText("● AI MICRO-EXPRESSION ANALYSIS", 48, 72);
  ctx.fillStyle = INK; ctx.font = `700 26px ${SANS}`;
  ctx.fillText(S.studio.title || "", 420, 74);
  ctx.textAlign = "right"; ctx.fillStyle = MUTED; ctx.font = `400 16px ${MONO}`;
  ctx.fillText(`FORENSIC SCIENCE · COURSE DEMO   ${t !== undefined ? fmtT(t) : ""}`, 1872, 72);
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(242,241,236,0.08)"; ctx.fillRect(48, 100, 1824, 1);
}

function drawStudioPanel(t) {
  const st = S.studio;
  topBar(t);
  // video frame border + caption
  ctx.strokeStyle = "rgba(242,241,236,0.12)"; ctx.lineWidth = 1;
  ctx.strokeRect(st.box.x + 0.5, st.box.y + 0.5, st.box.w - 1, st.box.h - 1);
  const m = markAt(t);
  if (m) {
    ctx.font = `700 24px ${SANS}`;
    const tw = ctx.measureText(m.text).width + 36;
    const x = st.box.x + 20, y = st.box.y + st.box.h - 64;
    roundRect(x, y, tw, 46, 23);
    ctx.fillStyle = m.kind === "key" ? "rgba(255,75,62,0.92)" : "rgba(7,7,8,0.82)"; ctx.fill();
    if (m.kind !== "key") { ctx.strokeStyle = LIME; ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.fillStyle = m.kind === "key" ? "#fff" : LIME; ctx.fillText(m.text, x + 18, y + 31);
  }
  if (!A.face) {
    ctx.textAlign = "center"; ctx.font = `700 26px ${SANS}`; ctx.fillStyle = AMBER;
    ctx.fillText("未偵測到臉部", st.box.x + st.box.w / 2, st.box.y + 50); ctx.textAlign = "left";
  }

  // right panel
  const px = 1268, pw = 604;
  roundRect(px, 128, pw, 704, 18); ctx.fillStyle = "#0e0f11"; ctx.fill();
  ctx.strokeStyle = "rgba(242,241,236,0.08)"; ctx.stroke();
  ctx.fillStyle = MUTED; ctx.font = `500 15px ${MONO}`; ctx.fillText("LIKELIHOOD INDEX", px + 28, 166);
  ctx.textAlign = "right"; ctx.fillStyle = A.calibrated ? LIME : BLUE; ctx.fillText(A.calibrated ? "● LIVE" : "● CALIBRATING", px + pw - 28, 166); ctx.textAlign = "left";
  gauge(px + pw / 2, 300, 112, 1, 20);
  ctx.textAlign = "center"; ctx.fillStyle = MUTED; ctx.font = `400 15px ${SANS}`;
  ctx.fillText("偏離受測者自身基準線的程度 · 非真實說謊機率", px + pw / 2, 436); ctx.textAlign = "left";

  let y = 476;
  ctx.fillStyle = MUTED; ctx.font = `500 14px ${MONO}`; ctx.fillText("CUES  (z vs baseline)", px + 28, y); y += 22;
  for (const k of KEYS) {
    const z = A.zs[k] || 0, hot = z > HOT;
    ctx.fillStyle = hot ? RED : INK; ctx.font = `500 19px ${SANS}`; ctx.fillText(CUES[k].zh, px + 28, y + 19);
    roundRect(px + 150, y + 9, 360, 8, 4); ctx.fillStyle = "rgba(242,241,236,0.08)"; ctx.fill();
    roundRect(px + 150, y + 9, Math.max(8, 360 * clamp(z / 4, 0, 1)), 8, 4); ctx.fillStyle = hot ? RED : LIME; ctx.fill();
    ctx.textAlign = "right"; ctx.fillStyle = MUTED; ctx.font = `400 15px ${MONO}`; ctx.fillText(z.toFixed(1), px + pw - 28, y + 19); ctx.textAlign = "left";
    y += 30;
  }
  // emotion + HR
  y += 8;
  ctx.fillStyle = "rgba(242,241,236,0.08)"; ctx.fillRect(px + 28, y, pw - 56, 1); y += 30;
  let top = "--";
  if (A.emo) { const i = A.emo.indexOf(Math.max(...A.emo)); top = `${EMO_ZH[i]} ${Math.round(A.emo[i] * 100)}%`; }
  ctx.fillStyle = MUTED; ctx.font = `500 14px ${MONO}`; ctx.fillText("EMOTION", px + 28, y); ctx.fillText("HEART RATE (rPPG)", px + 320, y);
  ctx.fillStyle = INK; ctx.font = `700 26px ${SANS}`; ctx.fillText(top, px + 28, y + 36);
  ctx.font = `600 26px ${DISPLAY}`; ctx.fillText(A.ppg.hr ? `${Math.round(A.ppg.hr)} bpm` : "--", px + 320, y + 36);

  // timeline (full clip)
  roundRect(48, 864, 1824, 176, 18); ctx.fillStyle = "#0e0f11"; ctx.fill();
  ctx.fillStyle = MUTED; ctx.font = `500 14px ${MONO}`; ctx.fillText("TIMELINE", 72, 896);
  ctx.textAlign = "right"; ctx.fillText(`${fmtT(st.start)} → ${fmtT(st.end)}`, 1848, 896); ctx.textAlign = "left";
  drawTimeline(72, 940, 1776, 84, st.start, st.end, 1, true, st.marks || []);
  const xp = 72 + ((t - st.start) / (st.end - st.start)) * 1776;
  ctx.fillStyle = INK; ctx.fillRect(xp, 914, 2, 112);
}

function drawIntro(e) {
  const st = S.studio;
  ctx.fillStyle = "#070708"; ctx.fillRect(0, 0, W, H);
  const a = clamp(e / 0.8, 0, 1) * clamp((INTRO - e) / 0.5, 0, 1);
  ctx.globalAlpha = a;
  ctx.fillStyle = LIME; ctx.font = `600 22px ${MONO}`; ctx.fillText(st.kicker || "CASE STUDY", 160, 380);
  ctx.fillStyle = INK; ctx.font = `900 92px ${SANS}`; ctx.fillText(st.title || "", 156, 500);
  ctx.fillStyle = MUTED; ctx.font = `400 32px ${SANS}`; ctx.fillText(st.sub || "", 160, 570);
  ctx.font = `400 20px ${MONO}`; ctx.fillText(st.source || "", 160, 680);
  ctx.fillStyle = BLUE; ctx.font = `500 24px ${SANS}`; ctx.fillText("前 10 秒建立受測者基準線（相當於測謊儀的控制問題）", 160, 740);
  ctx.globalAlpha = 1;
}

function drawOutro(e) {
  const st = S.studio, s = st.summary;
  const a = clamp(e / 0.6, 0, 1);
  ctx.fillStyle = `rgba(7,7,8,${0.12 * a})`; ctx.fillRect(0, 0, W, H); // accumulates into a fade over the last frame
  if (e < 0.6) return;
  ctx.fillStyle = "#070708"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = LIME; ctx.font = `600 22px ${MONO}`; ctx.fillText("ANALYSIS SUMMARY · 分析摘要", 160, 220);
  ctx.fillStyle = INK; ctx.font = `900 64px ${SANS}`; ctx.fillText(st.title || "", 156, 310);
  let x = 160;
  const stat = (label, val, color) => {
    ctx.fillStyle = MUTED; ctx.font = `500 22px ${SANS}`; ctx.fillText(label, x, 430);
    ctx.fillStyle = color; ctx.font = `600 96px ${DISPLAY}`; ctx.fillText(val, x, 530);
    x += 430;
  };
  for (const g of s.segs) stat(`${g.label}・平均`, g.avg === null ? "--" : `${Math.round(g.avg)}%`, g.kind === "key" ? RED : LIME);
  stat("全程峰值", `${Math.round(s.peak)}%`, AMBER);
  stat("偵測事件", `${s.events}`, INK);
  const top = Object.entries(s.counts).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, n]) => `${CUES[k].zh}×${n}`).join("   ");
  ctx.fillStyle = MUTED; ctx.font = `400 26px ${SANS}`; ctx.fillText(top ? `主要線索：${top}` : "主要線索：無", 160, 620);
  ctx.fillStyle = INK; ctx.font = `500 30px ${SANS}`;
  (st.note || "").split("|").forEach((line, i) => ctx.fillText(line, 160, 730 + i * 50));
  ctx.fillStyle = MUTED; ctx.font = `400 20px ${SANS}`;
  ctx.fillText("指標 = 臉部線索偏離個人基準線的程度。緊張、認知負荷也會使指標上升，不能單獨作為說謊證據。", 160, 920);
}

// ---------- events ----------
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (!el || el.disabled || el.tagName === "INPUT") return;
  const act = el.dataset.action;
  if (act === "cam") startCam();
  else if (act === "recal") { resetAnalysis(); if (S.mode === "cam") S.t0 = performance.now() / 1000; }
  else if (act === "rec") toggleRec();
  else if (act === "play") {
    if (video.paused || video.ended) { if (video.ended) video.currentTime = 0; video.play(); el.textContent = "❚❚ 暫停"; }
    else { video.pause(); el.textContent = "▶ 播放"; }
  }
});
document.querySelectorAll('input[data-action="file"]').forEach((inp) =>
  inp.addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) startFile(URL.createObjectURL(f));
    e.target.value = "";
  })
);

const q = new URLSearchParams(location.search);
loadModels().then(() => {
  if (!S.landmarker) return;
  if (q.get("studio")) {
    (q.get("offline") ? runStudioOffline : runStudio)({
      video: q.get("video"), start: +q.get("start") || 0, end: +q.get("end") || 60, name: q.get("name") || "demo",
      title: q.get("title") || "", sub: q.get("sub") || "", kicker: q.get("kicker") || "", source: q.get("source") || "",
      note: q.get("note") || "", marks: JSON.parse(q.get("marks") || "[]"),
    });
  } else if (q.get("video")) startFile(q.get("video"), true);
});
updatePanel();
