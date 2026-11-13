/* NSEQuant shared effects, all 2D canvas (no libraries).
   FX.loader({status})         a short light loader (logo line drawing itself, status, sliding bar) that fades out
                               once the page has its first data.
   FX.ready()                  tell the loader the page has its first data (it waits for this, 0.9 s to 4 s).
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
  // A short light loader: the logo's chart line draws itself, a status line and a sliding bar. It fades out once
  // the page has its first data (at least 0.9 s so it never flashes, at most 4 s so it never blocks).
  FX.loader = function (opt = {}) {
    if (REDUCE || document.hidden) { doReveal(); return; }
    const msgs = opt.status && opt.status.length ? opt.status : ["Connecting to the live feed"];
    const root = document.createElement("div");
    root.id = "fx-loader";
    root.setAttribute("role", "status");
    root.innerHTML = `<div class="fx-l">
      <svg class="fx-mark" viewBox="0 0 26 26" aria-hidden="true"><rect width="26" height="26" rx="6"/><path pathLength="1" d="M6 18 L10.5 8.5 L15 15 L20 7"/></svg>
      <div class="fx-word">NSE<span>Quant</span></div>
      <div class="fx-msg in">${msgs[0]}</div>
      <div class="fx-track"><i></i></div></div>`;
    document.body.appendChild(root);
    const msg = root.querySelector(".fx-msg"), t0 = performance.now();
    let i = 0;
    const cycle = setInterval(() => {
      if (i >= msgs.length - 1) return;
      msg.textContent = msgs[++i];
      msg.classList.remove("in"); void msg.offsetWidth; msg.classList.add("in");
    }, 650);
    (function wait() {
      const dt = performance.now() - t0;
      if ((readyFlag && dt > 900) || dt > 4000) {
        clearInterval(cycle);
        root.classList.add("out");
        doReveal();
        setTimeout(() => root.remove(), 450);
      } else setTimeout(wait, 100);                // a timer, not rAF: still fires in a background tab
    })();
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
