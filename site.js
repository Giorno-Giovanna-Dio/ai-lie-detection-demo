// Page chrome: hero data-field, scroll reveals, film tabs, counters.
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// --- hero field: a grid of vertical dashes bent by a travelling wave + the pointer ---
const cv = document.getElementById("field");
if (cv) {
  const c = cv.getContext("2d");
  let w = 0, h = 0, dpr = 1, mx = -9999, my = -9999, visible = true;
  const GX = 16, GY = 22;
  const resize = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  addEventListener("resize", resize);
  addEventListener("pointermove", (e) => { const r = cv.getBoundingClientRect(); mx = e.clientX - r.left; my = e.clientY - r.top; });
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(cv);
  const draw = (ms) => {
    requestAnimationFrame(draw);
    if (!visible) return;
    const t = reduce ? 0 : ms / 1000;
    c.clearRect(0, 0, w, h);
    const dim = new Path2D(), hot = new Path2D();
    for (let y = GY / 2; y < h; y += GY) {
      for (let x = GX / 2; x < w; x += GX) {
        const nx = x / w, ny = y / h;
        const wave = Math.sin(nx * 7 + t * 0.8) * Math.cos(ny * 5 - t * 0.5) + Math.sin((nx + ny) * 4 - t * 0.6) * 0.6;
        const d = Math.hypot(x - mx, y - my);
        const pull = Math.max(0, 1 - d / 220);
        const v = Math.max(0, wave * 0.5 + 0.35 + pull * 0.9);
        if (v < 0.42) continue;
        const len = 3 + v * 9;
        const tilt = (wave * 0.35 + pull * (x - mx) / 220) * 6;
        const path = pull > 0.35 || v > 1.05 ? hot : dim;
        path.moveTo(x - tilt / 2, y - len / 2); path.lineTo(x + tilt / 2, y + len / 2);
      }
    }
    c.lineWidth = 1.2; c.lineCap = "round";
    c.strokeStyle = "rgba(242,241,236,0.28)"; c.stroke(dim);
    c.strokeStyle = "rgba(215,255,58,0.85)"; c.stroke(hot);
  };
  requestAnimationFrame(draw);
}

// --- reveal on scroll ---
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
document.querySelectorAll(".reveal").forEach((el, i) => {
  el.style.transitionDelay = `${(i % 4) * 70}ms`;
  io.observe(el);
});

// --- count-up stats ---
const cio = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    cio.unobserve(e.target);
    const to = +e.target.dataset.count;
    if (reduce || !to) continue;
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / 1400);
      e.target.textContent = Math.round(to * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    };
    e.target.textContent = "0";
    requestAnimationFrame(step);
  }
}, { threshold: 0.6 });
document.querySelectorAll("[data-count]").forEach((el) => cio.observe(el));

// --- film tabs ---
const film = document.getElementById("film");
document.querySelectorAll(".film-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".film-tab").forEach((t) => t.classList.toggle("active", t === tab));
    film.poster = tab.dataset.src.replace("demo-", "poster-").replace(".mp4", ".jpg");
    film.src = tab.dataset.src;
    document.getElementById("filmCap").textContent = tab.dataset.cap;
  })
);
