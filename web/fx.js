/* NSEQuant shared effects, all 2D canvas (no libraries).
   FX.loader({title, status})  a glowing arrow chart climbs bottom-left to top-right, then the screen burns away from
                               the centre to reveal the page. Full intro once per browser session; later loads only burn.
   FX.ready()                  tell the loader the page has its first data (it waits for this, at most 4.5 s).
   FX.onReveal(fn)             run fn when the page is revealed (or at once if there is no loader).
   FX.rise(selector)           sections rise in as they scroll into view.     FX.tilt(selector)   tilt on hover.
   FX.count(el, to, fmt)       count-up.
   FX.depth(stageEl)           live 2D depth chart with fading history: .update(columns, tick)
   FX.curves(stageEl)          animated cumulative lines: .set(labels, {name: values}, {name: color}, fmt) */
(function () {
  const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const reveal = []; let revealed = false, readyFlag = false;
  const FX = {};
  FX.onReveal = fn => { revealed ? fn() : reveal.push(fn); };
  const doReveal = () => { if (revealed) return; revealed = true; reveal.splice(0).forEach(f => { try { f(); } catch (e) { console.error(e); } }); };
  FX.ready = () => { readyFlag = true; };

  function fitCanvas(cv, ctx) {     // CSS-pixel drawing on a device-pixel canvas
    const dpr = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }

  // ------------------------------------------------------------------------------------------------ loader
  FX.loader = function (opt = {}) {
    let seen = false;
    try { seen = sessionStorage.getItem("nsq_intro") === "1"; } catch (e) {}
    if (REDUCE || document.hidden) { doReveal(); return; }
    const short = seen;
    try { sessionStorage.setItem("nsq_intro", "1"); } catch (e) {}

    const root = document.createElement("div");
    root.id = "fx-loader";
    root.setAttribute("aria-label", "Loading");
    root.innerHTML = `<canvas></canvas><div class="fx-ui" ${short ? 'style="display:none"' : ""}>
      <div class="fx-brand"><svg viewBox="0 0 26 26"><rect width="26" height="26" rx="5" fill="#e6ebf5"/><path d="M6 18 L10.5 8.5 L15 15 L20 7" fill="none" stroke="#05070d" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>NSE<span>Quant</span></div>
      <div class="fx-title">${opt.title || "Microseconds, <i>compounded.</i>"}</div>
      <div class="fx-status" id="fx-status"></div>
      <div class="fx-count"><span id="fx-pct">0</span><small>%</small></div>
      <div class="fx-bar"><i id="fx-barfill"></i></div></div>
      <button id="fx-skip" type="button" ${short ? 'style="display:none"' : ""}>SKIP</button>`;
    document.body.appendChild(root);
    document.documentElement.style.overflow = "hidden";
    const STEPS = opt.status || ["Connecting to the live feed", "Loading the 532-byte int8 model", "Warming the hot path", "Rebuilding the order book", "Ready"];
    const cv = root.querySelector("canvas"), ctx = cv.getContext("2d");

    // deterministic price path, volume bars and sparks
    const N = 160, path = [];
    for (let i = 0; i <= N; i++) {
      const u = i / N;
      path.push(Math.max(0, Math.min(1, 0.04 + 0.92 * Math.pow(u, 1.25) + 0.055 * Math.sin(u * 23) * (1 - u * 0.4)
        + 0.03 * Math.sin(u * 57 + 1.3) - 0.07 * Math.exp(-((u - 0.55) ** 2) / 0.004))));
    }
    const NB = 46, bars = Array.from({ length: NB }, (_, i) => ({ u: (i + 0.5) / NB,
      h: 0.25 + 0.75 * Math.abs(Math.sin(i * 1.7) * 0.6 + Math.sin(i * 0.37) * 0.5), up: Math.sin(i * 2.3) > -0.35 }));
    const sparks = Array.from({ length: 140 }, () => ({ x: Math.random(), y: Math.random(), v: 0.0006 + Math.random() * 0.0016, r: 0.6 + Math.random() * 1.4 }));
    const NC = 64, candles = Array.from({ length: NC }, (_, i) => ({ u: (i + 0.5) / NC, up: Math.sin(i * 2.9 + 0.4) > -0.25,
      b: 0.012 + 0.03 * Math.abs(Math.sin(i * 1.3)), w: 0.006 + 0.016 * Math.abs(Math.cos(i * 0.9)) }));
    const embers = [];
    const phase = [Math.random() * 6.28, Math.random() * 6.28, Math.random() * 6.28];

    const t0 = performance.now(), MIN = short ? 0 : 3000;
    setTimeout(() => { readyFlag = true; }, short ? 0 : 4500);
    let prog = short ? 1 : 0, burnT = null, done = false, lastStep = -1;
    const pct = root.querySelector("#fx-pct"), barEl = root.querySelector("#fx-barfill"), status = root.querySelector("#fx-status");
    const skip = () => { prog = 1; readyFlag = true; if (burnT == null) startBurn(); };
    root.querySelector("#fx-skip").onclick = skip;
    addEventListener("keydown", function k(e) { if (e.key === "Escape") { skip(); removeEventListener("keydown", k); } });
    function startBurn() { burnT = performance.now(); root.classList.add("burning"); document.documentElement.style.overflow = ""; doReveal(); }

    function scene(W, H, p, t) {
      ctx.setTransform(Math.min(2, devicePixelRatio || 1), 0, 0, Math.min(2, devicePixelRatio || 1), 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#05070d"; ctx.fillRect(0, 0, W, H);
      const glow = ctx.createRadialGradient(W * 0.72, H * 0.25, 0, W * 0.72, H * 0.25, Math.max(W, H) * 0.7);
      glow.addColorStop(0, "rgba(37,72,160,0.32)"); glow.addColorStop(1, "rgba(5,7,13,0)");
      ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);

      const x0 = W * 0.05, x1 = W * 0.93, yb = H * 0.8, yt = H * 0.2, X = u => x0 + (x1 - x0) * u, Y = v => yb - (yb - yt) * v;
      const n = Math.max(2, Math.floor(p * N)), frac = p * N - Math.floor(p * N);
      const pts = path.slice(0, n + 1).map((v, i) => [X(i / N), Y(v)]);
      if (n < N) { const a = path[n], b = path[n + 1]; pts[n] = [X((n + frac) / N), Y(a + (b - a) * frac)]; }
      // camera: starts close on the tip at bottom-left, pulls back to the whole climb
      const [tx0, ty0] = pts[pts.length - 1], k = Math.pow(1 - p, 1.6), z = 1 + 1.15 * k;
      const fx = tx0 + (W / 2 - tx0) * (1 - k), fy = ty0 + (H / 2 - ty0) * (1 - k);
      const cam = (par) => { const zz = 1 + (z - 1) * par; ctx.translate(W / 2, H / 2); ctx.scale(zz, zz); ctx.translate(-(W / 2 + (fx - W / 2) * par), -(H / 2 + (fy - H / 2) * par)); };
      ctx.save(); cam(0.35);                                                  // far layer: grid and sparks move less (parallax)
      ctx.strokeStyle = "rgba(124,176,255,0.06)"; ctx.lineWidth = 1;
      const g = Math.max(48, Math.round(W / 26));
      for (let x = -W + (t * 6) % g; x < 2 * W; x += g) { ctx.beginPath(); ctx.moveTo(x, -H); ctx.lineTo(x, 2 * H); ctx.stroke(); }
      for (let y = -H; y < 2 * H; y += g) { ctx.beginPath(); ctx.moveTo(-W, y); ctx.lineTo(2 * W, y); ctx.stroke(); }
      sparks.forEach(s => { s.y -= s.v; if (s.y < 0) { s.y = 1; s.x = Math.random(); }
        ctx.fillStyle = `rgba(159,199,255,${0.25 + 0.5 * (1 - s.y)})`; ctx.beginPath(); ctx.arc(s.x * W, s.y * H, s.r, 0, 6.283); ctx.fill(); });
      ctx.restore();
      ctx.save(); cam(1);
      candles.forEach(c => {                                                  // candlesticks form along the line as it passes
        const kk = Math.max(0, Math.min(1, (p - c.u) * 9)); if (!kk) return;
        const v = path[Math.round(c.u * N)], cx = X(c.u), up = c.up, body = c.b * (yb - yt) * kk, wick = c.w * (yb - yt) * kk;
        const yo = Y(v) + (up ? body / 2 : -body / 2), yc = Y(v) + (up ? -body / 2 : body / 2), bw = (x1 - x0) / NC * 0.5;
        ctx.globalAlpha = 0.75 * kk; ctx.strokeStyle = ctx.fillStyle = up ? "#22d39a" : "#ff5a6a"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(cx, Math.min(yo, yc) - wick); ctx.lineTo(cx, Math.max(yo, yc) + wick); ctx.stroke();
        ctx.fillRect(cx - bw / 2, Math.min(yo, yc), bw, Math.max(1.5, Math.abs(yc - yo)));
        ctx.globalAlpha = 1;
      });
      bars.forEach(b => {                                                     // volume bars rise as the line passes
        const k = Math.max(0, Math.min(1, (p - b.u * 0.95) * 6)); if (!k) return;
        const bw = (x1 - x0) / NB * 0.58, bh = (H * 0.2) * b.h * k, bx = X(b.u) - bw / 2;
        const gr = ctx.createLinearGradient(0, yb - bh, 0, yb);
        gr.addColorStop(0, b.up ? "rgba(59,130,246,0.95)" : "rgba(234,57,67,0.85)"); gr.addColorStop(1, b.up ? "rgba(59,130,246,0.15)" : "rgba(234,57,67,0.12)");
        ctx.fillStyle = gr; ctx.fillRect(bx, yb - bh, bw, bh);
      });
      ctx.strokeStyle = "rgba(124,176,255,0.25)"; ctx.beginPath(); ctx.moveTo(x0, yb + 0.5); ctx.lineTo(x1, yb + 0.5); ctx.stroke();

      const area = ctx.createLinearGradient(0, yt, 0, yb);
      area.addColorStop(0, "rgba(59,130,246,0.42)"); area.addColorStop(1, "rgba(59,130,246,0)");
      ctx.beginPath(); ctx.moveTo(pts[0][0], yb); pts.forEach(q => ctx.lineTo(q[0], q[1])); ctx.lineTo(pts[pts.length - 1][0], yb); ctx.closePath();
      ctx.fillStyle = area; ctx.fill();
      const stroke = (w, c, blur) => { ctx.save(); ctx.lineJoin = ctx.lineCap = "round"; ctx.lineWidth = w; ctx.strokeStyle = c; ctx.shadowColor = "#3b82f6"; ctx.shadowBlur = blur;
        ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke(); ctx.restore(); };
      stroke(10, "rgba(59,130,246,0.25)", 30); stroke(3.2, "#cfe3ff", 18);
      const [tx, ty] = pts[pts.length - 1], [px, py] = pts[Math.max(0, pts.length - 4)], ang = Math.atan2(ty - py, tx - px);
      ctx.save(); ctx.translate(tx, ty); ctx.rotate(ang); ctx.fillStyle = "#ffffff"; ctx.shadowColor = "#7cb0ff"; ctx.shadowBlur = 24;   // arrowhead
      const s = Math.max(14, W / 80); ctx.beginPath(); ctx.moveTo(s * 1.25, 0); ctx.lineTo(-s * 0.55, -s * 0.75); ctx.lineTo(-s * 0.2, 0); ctx.lineTo(-s * 0.55, s * 0.75); ctx.closePath(); ctx.fill(); ctx.restore();
      const price = 1896.4 + 92.3 * path[Math.min(N, Math.round(p * N))], chg = (price / 1896.4 - 1) * 100;   // price tag riding the tip
      const lbl = `${price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}  ▲ ${chg.toFixed(2)}%`;
      ctx.font = `600 ${Math.max(11, Math.round(13 / z))}px 'IBM Plex Mono', monospace`; const lw = ctx.measureText(lbl).width + 16, lh = 22 / Math.sqrt(z);
      ctx.fillStyle = "rgba(230,235,245,0.95)"; ctx.fillRect(tx + 14, ty - lh - 10, lw, lh); ctx.fillStyle = "#05070d"; ctx.fillText(lbl, tx + 22, ty - 10 - lh * 0.32);
      ctx.restore();
      const vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);   // vignette
      vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.55)"); ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    }

    function blob(cx, cy, R, t) {           // a noisy circle: the burning edge
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const a = i / 200 * 6.2832, r = R * (1 + 0.13 * Math.sin(5 * a + phase[0] + t * 1.4) + 0.07 * Math.sin(11 * a + phase[1] - t * 2.1) + 0.035 * Math.sin(23 * a + phase[2] + t * 3));
        i ? ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)) : ctx.moveTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
      ctx.closePath();
    }

    function frame(now) {
      if (done) return;
      const [W, H] = fitCanvas(cv, ctx), t = (now - t0) / 1000;
      if (burnT == null) {
        const target = Math.min(1, (now - t0) / Math.max(1, MIN)) * (readyFlag ? 1 : 0.92);
        prog += (target - prog) * (short ? 1 : 0.08);
        if (prog > 0.995 && readyFlag && now - t0 > MIN) { prog = 1; startBurn(); }
      }
      const p = Math.min(1, prog);
      scene(W, H, p, t);
      if (pct) pct.textContent = Math.round(p * 100);
      if (barEl) barEl.style.width = (p * 100).toFixed(1) + "%";
      const st = Math.min(STEPS.length - 1, Math.floor(p * STEPS.length));
      if (status && st !== lastStep) { lastStep = st; status.innerHTML = STEPS.map((x, i) => i < st ? `${x} ✓` : i === st ? `<b>${x}</b>` : "").filter(Boolean).slice(-3).join("<br>"); }
      if (burnT != null) {
        const b = Math.min(1, (now - burnT) / (short ? 750 : 1300)), e = b * b * (3 - 2 * b);
        const R = Math.hypot(W, H) * 0.62 * e, cx = W / 2, cy = H / 2;
        if (b >= 1) { done = true; root.remove(); return; }
        ctx.save(); blob(cx, cy, R + 38, t); ctx.lineWidth = 70; ctx.strokeStyle = "rgba(25,10,2,0.55)"; ctx.stroke(); ctx.restore();   // char
        ctx.save(); ctx.globalCompositeOperation = "destination-out"; blob(cx, cy, R, t); ctx.fill(); ctx.restore();                      // hole
        ctx.save(); ctx.lineJoin = "round";                                                                                              // burning edge
        [[22, "rgba(255,72,10,0.35)", 40], [9, "rgba(255,120,20,0.9)", 26], [3.5, "rgba(255,226,150,1)", 12]].forEach(([w, c, bl]) => {
          blob(cx, cy, R + 2, t); ctx.lineWidth = w; ctx.strokeStyle = c; ctx.shadowColor = "#ff6a00"; ctx.shadowBlur = bl; ctx.stroke(); });
        ctx.restore();
        for (let i = 0; i < 26 && R > 4; i++) {                                     // embers thrown off the edge
          const a = Math.random() * 6.2832, sp = 1.2 + Math.random() * 3.2;
          embers.push({ x: cx + (R + 4) * Math.cos(a), y: cy + (R + 4) * Math.sin(a), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 1.2, life: 1, r: 0.8 + Math.random() * 2.2 });
        }
        ctx.save(); ctx.globalCompositeOperation = "lighter";
        for (let i = embers.length - 1; i >= 0; i--) {
          const m = embers[i]; m.x += m.vx; m.y += m.vy; m.vy -= 0.04; m.vx *= 0.985; m.life -= 0.02 + Math.random() * 0.012;
          if (m.life <= 0) { embers.splice(i, 1); continue; }
          ctx.fillStyle = `rgba(255,${Math.round(120 + 120 * m.life)},${Math.round(40 * m.life)},${m.life})`;
          ctx.shadowColor = "#ff7a1a"; ctx.shadowBlur = 10; ctx.beginPath(); ctx.arc(m.x, m.y, m.r * (0.5 + m.life), 0, 6.283); ctx.fill();
        }
        if (b < 0.18) { const fl = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.5);   // ignition flash
          fl.addColorStop(0, `rgba(255,220,170,${0.55 * (1 - b / 0.18)})`); fl.addColorStop(1, "rgba(255,120,30,0)"); ctx.fillStyle = fl; ctx.fillRect(0, 0, W, H); }
        ctx.restore();
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  };

  // ------------------------------------------------------------------------------------------------ page effects
  FX.rise = function (sel) {
    const els = [...document.querySelectorAll(sel)];
    els.forEach((el, i) => { el.classList.add("fx-rise"); el.style.setProperty("--d", (Math.min(i, 6) * 0.06) + "s"); });
    if (REDUCE || !("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
    FX.onReveal(() => {
      const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
      els.forEach(e => io.observe(e));
    });
  };
  FX.tilt = function (sel, max = 5) {
    if (REDUCE || matchMedia("(hover: none)").matches) return;
    document.querySelectorAll(sel).forEach(el => {
      el.classList.add("fx-tilt");
      el.addEventListener("pointermove", e => {
        const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform = `perspective(900px) rotateX(${(-y * max).toFixed(2)}deg) rotateY(${(x * max).toFixed(2)}deg) translateZ(4px)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  };
  FX.count = function (el, to, fmt, dur = 1600) {
    if (to == null || !isFinite(to)) return;
    if (REDUCE || document.hidden) { el.innerHTML = fmt(to); return; }
    const from = el._fxv == null ? 0 : el._fxv, t0 = performance.now(); el._fxv = to;
    const step = now => { const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 4); el.innerHTML = fmt(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  };

  function stage(el, draw) {       // a canvas in a dark stage, redrawn every frame while visible
    const cv = document.createElement("canvas"); el.prepend(cv);
    const ctx = cv.getContext("2d");
    let visible = true, raf = 0;
    const loop = now => { raf = 0; if (!visible || document.hidden) return; const [w, h] = fitCanvas(cv, ctx); if (w && h) draw(ctx, w, h, now); raf = requestAnimationFrame(loop); };
    new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && !raf) raf = requestAnimationFrame(loop); }).observe(el);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && visible && !raf) raf = requestAnimationFrame(loop); });
    raf = requestAnimationFrame(loop);
    return cv;
  }
  const MONO = "500 11px 'IBM Plex Mono', monospace";

  // ------------------------------------------------------------------------------------------------ live 2D depth chart
  FX.depth = function (el, W = 40) {
    /* state arrays: [bid cumulative at offsets -W..0, ask cumulative at offsets 0..W] (length 2W+2) */
    let cur = null, target = null, hist = [], lastHist = 0, info = null;
    stage(el, (ctx, w, h, now) => {
      const g = ctx.createLinearGradient(0, 0, 0, h); g.addColorStop(0, "#0d1730"); g.addColorStop(1, "#05070d");
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      if (!target) { ctx.fillStyle = "#8f9bb3"; ctx.font = MONO; ctx.fillText("waiting for the order book…", 18, h / 2); return; }
      if (!cur) cur = target.slice();
      for (let i = 0; i < cur.length; i++) cur[i] += (target[i] - cur[i]) * 0.15;
      if (now - lastHist > 1200) { hist.push(cur.slice()); if (hist.length > 10) hist.shift(); lastHist = now; }
      const L = 56, Rr = 16, T = 48, B = 46, cw = w - L - Rr, ch = h - T - B;
      const mx = Math.max(1e-9, ...cur, ...hist.flat()) * 1.08;
      const X = off => L + cw * ((off + W) / (2 * W)), Y = v => T + ch - ch * (v / mx);
      const side = (arr, sd) => sd > 0 ? Array.from({ length: W + 1 }, (_, i) => [X(i - W), Y(arr[i])]) : Array.from({ length: W + 1 }, (_, j) => [X(j), Y(arr[W + 1 + j])]);
      const path = pts => { ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); };
      ctx.strokeStyle = "rgba(124,176,255,0.08)"; ctx.lineWidth = 1;          // grid
      for (let k = 0; k <= 4; k++) { const y = T + ch * k / 4; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + cw, y); ctx.stroke(); }
      ctx.fillStyle = "#8f9bb3"; ctx.font = MONO; ctx.textAlign = "right";
      for (let k = 0; k <= 4; k++) { const v = mx * (1 - k / 4); ctx.fillText(v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0), L - 8, T + ch * k / 4 + 4); }
      hist.forEach((hh, j) => { const a = 0.06 + 0.22 * (j / hist.length);           // fading history: where the book was
        [1, -1].forEach(sd => { path(side(hh, sd)); ctx.strokeStyle = sd > 0 ? `rgba(22,199,132,${a})` : `rgba(234,57,67,${a})`; ctx.lineWidth = 1.2; ctx.stroke(); }); });
      [1, -1].forEach(sd => {
        const pts = side(cur, sd);
        path(pts); ctx.lineTo(pts[pts.length - 1][0], T + ch); ctx.lineTo(pts[0][0], T + ch); ctx.closePath();
        const gr = ctx.createLinearGradient(0, T, 0, T + ch);
        gr.addColorStop(0, sd > 0 ? "rgba(22,199,132,0.55)" : "rgba(234,57,67,0.55)"); gr.addColorStop(1, sd > 0 ? "rgba(22,199,132,0.04)" : "rgba(234,57,67,0.04)");
        ctx.fillStyle = gr; ctx.fill();
        path(pts); ctx.save(); ctx.lineWidth = 2.2; ctx.strokeStyle = sd > 0 ? "#3ee6a4" : "#ff6b74"; ctx.shadowColor = sd > 0 ? "#16c784" : "#ea3943"; ctx.shadowBlur = 14; ctx.stroke(); ctx.restore();
      });
      const xm = X(0), pulse = 0.5 + 0.5 * Math.sin(now / 350);
      ctx.strokeStyle = `rgba(230,235,245,${0.35 + 0.35 * pulse})`; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(xm, T - 6); ctx.lineTo(xm, T + ch); ctx.stroke(); ctx.setLineDash([]);
      if (info) {
        ctx.font = "600 12px 'IBM Plex Mono', monospace"; ctx.textAlign = "center"; const tw = ctx.measureText(info.mid).width + 14;
        ctx.fillStyle = "#e6ebf5"; ctx.fillRect(xm - tw / 2, T - 26, tw, 19); ctx.fillStyle = "#05070d"; ctx.fillText(info.mid, xm, T - 12);
        ctx.font = MONO; ctx.fillStyle = "#8f9bb3"; ctx.textAlign = "left"; ctx.fillText(info.lo, L, T + ch + 18); ctx.textAlign = "right"; ctx.fillText(info.hi, L + cw, T + ch + 18);
        ctx.textAlign = "left"; ctx.fillStyle = "#3ee6a4"; ctx.fillText(`bids ${info.bid}`, L, 24); ctx.fillStyle = "#ff6b74"; ctx.textAlign = "right"; ctx.fillText(`asks ${info.ask}`, L + cw, 24);
      }
    });
    return {
      /* columns: [ts, lowestTick, bidTick, askTick, sizes[]], oldest first; tick = price per tick */
      update(cols, tick = 0.01) {
        if (!cols || !cols.length) return;
        const [, base, bid, ask, q] = cols[cols.length - 1], mid = Math.round((bid + ask) / 2);
        const out = new Array(2 * W + 2).fill(0);
        let cum = 0; for (let i = W; i >= 0; i--) { const tk = mid - W + i; if (tk <= bid) cum += q[tk - base] || 0; out[i] = cum; }
        cum = 0; for (let j = 0; j <= W; j++) { const tk = mid + j; if (tk >= ask) cum += q[tk - base] || 0; out[W + 1 + j] = cum; }
        if (target && target.length !== out.length) { cur = null; hist = []; }
        target = out;
        const dec = Math.max(0, -Math.floor(Math.log10(tick)));
        const f = v => (v * tick).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
        info = { mid: f((bid + ask) / 2), lo: f(mid - W), hi: f(mid + W), bid: out[0].toFixed(1), ask: out[2 * W + 1].toFixed(1) };
      },
    };
  };

  // ------------------------------------------------------------------------------------------------ animated cumulative lines
  FX.curves = function (el) {
    let data = null, t0 = 0;
    stage(el, (ctx, w, h, now) => {
      const g = ctx.createLinearGradient(0, 0, 0, h); g.addColorStop(0, "#0d1730"); g.addColorStop(1, "#05070d");
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      if (!data) return;
      const { labels, series, colors, fmt } = data, names = Object.keys(series), n = labels.length;
      const all = names.flatMap(k => series[k]).concat([0]);
      const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.12 || 1;
      const L = 64, Rr = 130, T = 44, B = 50, cw = w - L - Rr, ch = h - T - B;
      const X = i => L + cw * (n > 1 ? i / (n - 1) : 0.5), Y = v => T + ch - ch * ((v - (lo - pad)) / ((hi + pad) - (lo - pad)));
      const p = REDUCE ? 1 : Math.min(1, (now - t0) / 1800), e = 1 - Math.pow(1 - p, 3);
      ctx.font = MONO; ctx.lineWidth = 1;
      for (let k = 0; k <= 4; k++) { const v = (lo - pad) + ((hi + pad) - (lo - pad)) * k / 4, y = Y(v);
        ctx.strokeStyle = "rgba(124,176,255,0.08)"; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + cw, y); ctx.stroke();
        ctx.fillStyle = "#8f9bb3"; ctx.textAlign = "right"; ctx.fillText(Math.round(v).toLocaleString("en-US"), L - 8, y + 4); }
      ctx.strokeStyle = "rgba(230,235,245,0.45)"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(L + cw, Y(0)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = "center"; ctx.fillStyle = "#8f9bb3"; labels.forEach((l, i) => ctx.fillText(l, X(i), T + ch + 20));
      const upto = e * (n - 1);
      const ends = [];
      names.forEach(k => {
        const v = series[k], col = colors[k];
        const pts = []; for (let i = 0; i <= Math.floor(upto); i++) pts.push([X(i), Y(v[i])]);
        const fi = Math.floor(upto); if (fi < n - 1) { const fr = upto - fi; pts.push([X(fi + fr), Y(v[fi] + (v[fi + 1] - v[fi]) * fr)]); }
        if (k === names[0]) {                                        // area under the headline series
          const gr = ctx.createLinearGradient(0, T, 0, T + ch); gr.addColorStop(0, col + "55"); gr.addColorStop(1, col + "00");
          ctx.beginPath(); ctx.moveTo(pts[0][0], Y(0)); pts.forEach(q => ctx.lineTo(q[0], q[1])); ctx.lineTo(pts[pts.length - 1][0], Y(0)); ctx.closePath(); ctx.fillStyle = gr; ctx.fill();
        }
        ctx.save(); ctx.lineJoin = ctx.lineCap = "round"; ctx.strokeStyle = col; ctx.lineWidth = k === names[0] ? 3 : 2; ctx.shadowColor = col; ctx.shadowBlur = k === names[0] ? 16 : 8;
        ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke(); ctx.restore();
        pts.slice(0, Math.floor(upto) + 1).forEach(q => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(q[0], q[1], 3.2, 0, 6.283); ctx.fill(); });
        const last = pts[pts.length - 1], val = v[Math.min(n - 1, Math.round(upto))];
        ends.push({ y: last[1], x: last[0], k, col, txt: `${k}  ${fmt(p >= 1 ? v[n - 1] : val)}` });
      });
      ends.sort((a, b) => a.y - b.y);                                // end labels, nudged apart
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
      ctx.font = "600 12px 'IBM Plex Mono', monospace"; ctx.textAlign = "left";
      ends.forEach(o => { ctx.fillStyle = o.col; ctx.beginPath(); ctx.arc(o.x, o.y, 5, 0, 6.283); ctx.fill(); ctx.fillStyle = "#e6ebf5"; ctx.fillText(o.txt, o.x + 12, o.y + 4); });
    });
    return { set(labels, series, colors, fmt = v => v.toFixed(2)) { data = { labels, series, colors, fmt }; t0 = performance.now(); } };
  };

  // ------------------------------------------------------------------------------------------------ kinetic details
  /* scrolling ticker band: FX.marquee(el).set([[label, value, cls], ...]) */
  FX.marquee = function (el) {
    el.classList.add("fx-marquee"); el.setAttribute("aria-hidden", "true");
    el.innerHTML = '<div class="fx-mq-track"><div class="fx-mq-run"></div><div class="fx-mq-run"></div></div>';
    const runs = el.querySelectorAll(".fx-mq-run");
    return { set(items) {
      const html = items.map(([l, v, c]) => `<span class="fx-mq-item"><b>${l}</b><em class="${c || ""}">${v}</em></span><span class="fx-mq-sep">✦</span>`).join("");
      runs.forEach(r => { if (r.innerHTML !== html) r.innerHTML = html; });
    } };
  };
  /* headline words rise into place one by one when the page is revealed */
  FX.words = function (sel) {
    document.querySelectorAll(sel).forEach(h => {
      let i = 0;
      const walk = node => [...node.childNodes].forEach(c => {
        if (c.nodeType === 3) {
          const frag = document.createDocumentFragment();
          c.textContent.split(/(\s+)/).forEach(w => {
            if (!w.trim()) { frag.append(w); return; }
            const o = document.createElement("span"); o.className = "fx-w";
            const s = document.createElement("span"); s.textContent = w; s.style.setProperty("--i", i++); o.append(s); frag.append(o);
          });
          c.replaceWith(frag);
        } else if (c.nodeType === 1) walk(c);
      });
      walk(h);
      h.classList.add("fx-words");
      FX.onReveal(() => requestAnimationFrame(() => h.classList.add("in")));
    });
  };
  /* a ring that trails the mouse and swells over anything clickable (fine pointers only) */
  FX.cursor = function () {
    if (REDUCE || !matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    const r = document.createElement("div"); r.className = "fx-cursor"; document.body.appendChild(r);
    let x = -100, y = -100, cx = x, cy = y;
    addEventListener("pointermove", e => { x = e.clientX; y = e.clientY;
      r.classList.toggle("big", !!e.target.closest("a, button, input, [role=button], .tseg button, .seg button")); }, { passive: true });
    addEventListener("pointerdown", () => r.classList.add("down")); addEventListener("pointerup", () => r.classList.remove("down"));
    document.addEventListener("pointerleave", () => r.classList.add("gone")); document.addEventListener("pointerenter", () => r.classList.remove("gone"));
    (function loop() { cx += (x - cx) * 0.22; cy += (y - cy) * 0.22; r.style.transform = `translate(${cx}px, ${cy}px)`; requestAnimationFrame(loop); })();
  };
  /* a soft light that follows the mouse across cards */
  FX.spotlight = function (sel) {
    if (REDUCE || !matchMedia("(hover: hover)").matches) return;
    document.querySelectorAll(sel).forEach(el => {
      el.classList.add("fx-spot");
      el.addEventListener("pointermove", e => { const b = el.getBoundingClientRect(); el.style.setProperty("--mx", (e.clientX - b.left) + "px"); el.style.setProperty("--my", (e.clientY - b.top) + "px"); });
    });
  };

  window.FX = FX;
})();
