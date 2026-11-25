// AI 微表情測謊分析 Demo
// Browser equivalent of the OpenCV + DeepFace pipeline (see python/analyze_video.py):
//   frames → MediaPipe Face Landmarker (landmarks + blendshapes ≈ FACS action units)
//          → face-api expression net (7 emotions, same classes as DeepFace)
//          → per-subject baseline (first CAL_SECONDS) → z-scores → weighted score 0–100.

import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACEAPI_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/dist/face-api.esm.js";
const FACEAPI_MODELS = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/";

const CAL_SECONDS = 10;

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

// ---------- state ----------
const S = {
  landmarker: null, faceapi: null, draw: null,
  mode: null, mirrored: false, running: false, t0: 0,
  rec: null, recStart: 0, camStream: null,
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
const scoreColor = (s) => (s < 40 ? "#5cf2d6" : s < 70 ? "#ffb547" : "#ff4d6d");
const clock = () => (S.mode === "file" ? video.currentTime : performance.now() / 1000 - S.t0);

function setStatus(text, kind) {
  $("#statusText").textContent = text;
  $("#status").className = `status ${kind}`;
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
    S.draw = new DrawingUtils(ctx);
  } catch (e) {
    console.error(e);
    setStatus("臉部模型載入失敗，請檢查網路後重新整理", "error");
    return;
  }
  setStatus("臉部模型就緒 · 載入情緒模型中…", "loading");
  enableSources();
  try {
    const fa = await import(FACEAPI_URL);
    try { await fa.tf.setBackend("webgl"); } catch { /* fall back to default backend */ }
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
  video.loop = false;
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
  if (S.rec) S.rec.stop();
  $("#playBtn").textContent = "▶ 重播";
});

// ---------- main loop ----------
function loop() {
  requestAnimationFrame(loop);
  if (!S.running || !S.landmarker || video.readyState < 2) return;
  if (S.mode === "file" && video.currentTime === A.lastVideoTime) return; // paused / no new frame
  A.lastVideoTime = video.currentTime;
  try { processFrame(); } catch (e) { console.error(e); }
}
requestAnimationFrame(loop);

function fitCanvas() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;
  const scale = Math.min(1, 1280 / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  W = w; H = h;
}

function processFrame() {
  const t = clock();
  if (A.lastT !== null && t < A.lastT - 0.5) resetAnalysis(); // video was rewound
  const dt = A.lastT === null ? 1 / 30 : clamp(t - A.lastT, 1e-3, 0.25);
  A.lastT = t;

  const res = S.landmarker.detectForVideo(video, performance.now());
  fitCanvas();
  ctx.save();
  if (S.mirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

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
    A.score = ema(A.score, A.calibrated ? 0 : 0, dt, 1.5);
  }
  if (A.calibrated) A.timeline.push({ t, s: A.score });
  if (A.timeline.length > 3000) A.timeline.splice(0, A.timeline.length - 3000);
  drawHUD(t);
  if (t - A.lastUI > 0.1 || t < A.lastUI) { A.lastUI = t; updatePanel(); }
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
  // detrend with a 1.5 s moving average, normalise, Hann window
  const pre = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) pre[i + 1] = pre[i] + x[i];
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) { const lo = Math.max(0, i - 22), hi = Math.min(N - 1, i + 22); y[i] = x[i] - (pre[hi + 1] - pre[lo]) / (hi - lo + 1); }
  const { std } = meanStd(Array.from(y));
  if (!std) return;
  for (let i = 0; i < N; i++) y[i] = (y[i] / std) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
  // power spectrum over 45–180 bpm
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
    // event log on upward crossing
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
function drawFace(lm) {
  const du = S.draw;
  const lw = Math.max(1, W / 800);
  const col = (k) => (A.zs[k] > HOT ? "#ff4d6d" : "#5cf2d6");
  ctx.save();
  if (S.mirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(92,242,214,0.10)", lineWidth: 0.5 });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "rgba(92,242,214,0.45)", lineWidth: lw });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: col("brow"), lineWidth: lw * 2 });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: col("brow"), lineWidth: lw * 2 });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: col("blink"), lineWidth: lw });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: col("blink"), lineWidth: lw });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: col("gaze"), lineWidth: lw * 1.5 });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: col("gaze"), lineWidth: lw * 1.5 });
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: col("lip"), lineWidth: lw * 1.5 });
  ctx.restore();

  // corner-bracket box around the face
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of lm) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  if (S.mirrored) [x0, x1] = [1 - x1, 1 - x0];
  const pad = 0.04;
  const bx = (x0 - pad) * W, by = (y0 - pad) * H, bw = (x1 - x0 + 2 * pad) * W, bh = (y1 - y0 + 2 * pad) * H;
  const c = Math.min(bw, bh) * 0.15;
  ctx.strokeStyle = A.calibrated ? scoreColor(A.score) : "#6aa8ff";
  ctx.lineWidth = lw * 2;
  ctx.beginPath();
  [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]].forEach(([x, y, sx, sy]) => {
    ctx.moveTo(x + sx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * c);
  });
  ctx.stroke();
  const u = Math.min(W, H) / 720;
  ctx.font = `600 ${13 * u}px "JetBrains Mono", monospace`;
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fillText("SUBJECT 01", bx, by - 8 * u);
}

function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }

function drawHUD(t) {
  const u = Math.min(W, H) / 720;
  const sans = (px, wt = 700) => `${wt} ${px * u}px "Noto Sans TC", "PingFang TC", system-ui, sans-serif`;
  const mono = (px, wt = 600) => `${wt} ${px * u}px "JetBrains Mono", ui-monospace, monospace`;

  // top-left banner
  roundRect(16 * u, 16 * u, 330 * u, 62 * u, 10 * u);
  ctx.fillStyle = "rgba(5,10,15,0.72)"; ctx.fill();
  ctx.fillStyle = "#5cf2d6"; ctx.font = mono(12);
  ctx.fillText("AI MICRO-EXPRESSION ANALYSIS", 30 * u, 40 * u);
  ctx.fillStyle = "#e6edf5"; ctx.font = mono(16, 400);
  ctx.fillText(fmtT(Math.max(0, t)), 30 * u, 64 * u);
  if (S.rec) {
    const on = Math.floor(performance.now() / 500) % 2 === 0;
    ctx.fillStyle = on ? "#ff4d6d" : "rgba(255,77,109,0.4)";
    ctx.beginPath(); ctx.arc(250 * u, 59 * u, 6 * u, 0, Math.PI * 2); ctx.fill();
    ctx.font = mono(13); ctx.fillText("REC", 262 * u, 64 * u);
  }

  // gauge (top-right)
  const R = 78 * u, gx = W - R - 34 * u, gy = R + 34 * u;
  ctx.fillStyle = "rgba(5,10,15,0.72)";
  ctx.beginPath(); ctx.arc(gx, gy, R + 18 * u, 0, Math.PI * 2); ctx.fill();
  const a0 = Math.PI * 0.75, span = Math.PI * 1.5;
  ctx.lineCap = "round"; ctx.lineWidth = 12 * u;
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span); ctx.stroke();
  ctx.textAlign = "center";
  if (A.calibrated) {
    const sc = Math.round(A.score);
    ctx.strokeStyle = scoreColor(A.score);
    ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span * (A.score / 100)); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = mono(46);
    ctx.fillText(`${sc}%`, gx, gy + 12 * u);
    ctx.fillStyle = scoreColor(A.score); ctx.font = sans(13);
    ctx.fillText("說謊可能性", gx, gy + 40 * u);
  } else {
    const p = A.calStart === null ? 0 : clamp((t - A.calStart) / CAL_SECONDS, 0, 1);
    ctx.strokeStyle = "#6aa8ff";
    ctx.beginPath(); ctx.arc(gx, gy, R, a0, a0 + span * p); ctx.stroke();
    ctx.fillStyle = "#6aa8ff"; ctx.font = sans(17);
    ctx.fillText("建立基準線", gx, gy + 2 * u);
    ctx.font = mono(13, 400); ctx.fillStyle = "#e6edf5";
    ctx.fillText(`${Math.ceil(CAL_SECONDS * (1 - p))}s`, gx, gy + 26 * u);
  }
  ctx.textAlign = "left"; ctx.lineCap = "butt";

  // active cue chips under the gauge
  if (A.calibrated) {
    const hot = KEYS.filter((k) => A.zs[k] > HOT).sort((a, b) => A.zs[b] - A.zs[a]).slice(0, 4);
    let cy = gy + R + 34 * u;
    ctx.font = sans(14, 500);
    for (const k of hot) {
      const label = `▲ ${CUES[k].zh}`;
      const tw = ctx.measureText(label).width + 20 * u;
      const x = W - 16 * u - tw;
      roundRect(x, cy, tw, 28 * u, 14 * u);
      ctx.fillStyle = "rgba(255,77,109,0.88)"; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(label, x + 10 * u, cy + 19 * u);
      cy += 34 * u;
    }
  }

  // bottom timeline
  const th = 70 * u, tx = 16 * u, tw = W - 32 * u, ty = H - th - 16 * u;
  roundRect(tx, ty, tw, th, 10 * u);
  ctx.fillStyle = "rgba(5,10,15,0.72)"; ctx.fill();
  ctx.font = mono(10, 400); ctx.fillStyle = "#8a9bb0";
  ctx.fillText("LIKELIHOOD · LAST 30s", tx + 10 * u, ty + 15 * u);
  const px0 = tx + 10 * u, pw = tw - 20 * u, py0 = ty + 22 * u, ph = th - 30 * u;
  const yOf = (s) => py0 + ph - (s / 100) * ph;
  ctx.setLineDash([4 * u, 4 * u]); ctx.strokeStyle = "rgba(255,77,109,0.5)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px0, yOf(70)); ctx.lineTo(px0 + pw, yOf(70)); ctx.stroke(); ctx.setLineDash([]);
  const tl = A.timeline;
  if (tl.length > 1) {
    const tEnd = tl[tl.length - 1].t, tStart = tEnd - 30;
    const xOf = (tt) => px0 + ((tt - tStart) / 30) * pw;
    ctx.beginPath();
    let started = false;
    for (const p of tl) { if (p.t < tStart) continue; const x = xOf(p.t), y = yOf(p.s); started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), (started = true)); }
    ctx.strokeStyle = scoreColor(A.score); ctx.lineWidth = 2 * u; ctx.stroke();
    for (const ev of A.events) { if (ev.t < tStart) break; ctx.fillStyle = "#ff4d6d"; ctx.beginPath(); ctx.arc(xOf(ev.t), py0 + 2 * u, 3 * u, 0, Math.PI * 2); ctx.fill(); }
  }

  if (!A.face) {
    ctx.textAlign = "center"; ctx.font = sans(20);
    ctx.fillStyle = "rgba(255,181,71,0.95)";
    ctx.fillText("未偵測到臉部 · 請面向鏡頭", W / 2, H / 2);
    ctx.textAlign = "left";
  }
}

// ---------- side panel ----------
const cueList = $("#cues");
cueList.innerHTML = KEYS.map((k) => `<li data-k="${k}"><span>${CUES[k].zh}</span><span class="bar"><i></i></span><span class="v">0.0</span></li>`).join("");
const emoList = $("#emos");
emoList.innerHTML = EMO_ZH.map((n) => `<li><span>${n}</span><span class="bar"><i></i></span><span class="v">0</span></li>`).join("");

function updatePanel() {
  const phase = $("#phase");
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
  const ol = $("#events");
  if (!A || !A.events.length) { ol.innerHTML = '<li class="empty">尚無事件</li>'; return; }
  ol.innerHTML = A.events.slice(0, 40).map((e) => `<li><span class="t">${fmtT(e.t)}</span><span>${CUES[e.k].zh}</span></li>`).join("");
}

// ---------- recording ----------
function toggleRec() {
  if (S.rec) { S.rec.stop(); return; }
  const stream = canvas.captureStream(30);
  try {
    if (S.mode === "cam" && S.camStream) S.camStream.getAudioTracks().forEach((tr) => stream.addTrack(tr));
    else if (S.mode === "file") {
      const vs = video.captureStream ? video.captureStream() : video.mozCaptureStream ? video.mozCaptureStream() : null;
      if (vs) vs.getAudioTracks().forEach((tr) => stream.addTrack(tr));
    }
  } catch { /* video only */ }
  const mime = ["video/mp4;codecs=avc1.42E01E,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm"].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  let rec;
  try { rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000 } : {}); }
  catch (e) { alert("此瀏覽器不支援錄影：" + e.message); return; }
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const type = rec.mimeType || "video/webm";
    const blob = new Blob(chunks, { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ai-lie-detection-demo-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${type.includes("mp4") ? "mp4" : "webm"}`;
    document.body.appendChild(a); a.click(); a.remove();
    S.rec = null;
    $("#recBtn").classList.remove("on");
    $("#recBtn").textContent = "● 錄製示範影片";
  };
  rec.start(1000);
  S.rec = rec;
  $("#recBtn").classList.add("on");
  $("#recBtn").textContent = "■ 停止並下載";
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
    if (video.paused || video.ended) { if (video.ended) { video.currentTime = 0; } video.play(); el.textContent = "❚❚ 暫停"; }
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

// Optional: ?video=<url> loads a remote clip (needs CORS for heart-rate).
const qv = new URLSearchParams(location.search).get("video");

loadModels().then(() => { if (qv && S.landmarker) startFile(qv, true); });
updatePanel();
