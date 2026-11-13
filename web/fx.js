/* NSEQuant shared effects (needs three.js r128 as window.THREE; everything degrades gracefully without it).
   FX.loader({title, status})  3D rising-arrow chart, then the screen burns away from the centre to reveal the page.
                               Full intro once per browser session; later page loads get only the burn.
   FX.ready()                  tell the loader the page has its first data (it waits for this, at most 4.5 s).
   FX.onReveal(fn)             run fn when the page is revealed (or at once if there is no loader).
   FX.rise(selector)           sections rise in as they scroll into view.     FX.tilt(selector)   3D tilt on hover.
   FX.count(el, to, fmt)       count-up.
   FX.terrain(stageEl)         live 3D liquidity terrain (price x time x resting size): .update(columns, tick)
   FX.bars3d(stageEl)          3D bars of a value per (category, series): .set(categories, series, colors) */
(function () {
  const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const HAS3D = () => typeof THREE !== "undefined" && (() => { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; } })();
  const reveal = []; let revealed = false, readyFlag = false;
  const FX = {};
  FX.onReveal = fn => { revealed ? fn() : reveal.push(fn); };
  const doReveal = () => { if (revealed) return; revealed = true; reveal.splice(0).forEach(f => { try { f(); } catch (e) { console.error(e); } }); };
  FX.ready = () => { readyFlag = true; };

  // ------------------------------------------------------------------------------------------------ loader
  FX.loader = function (opt = {}) {
    let seen = false;
    try { seen = sessionStorage.getItem("nsq_intro") === "1"; } catch (e) {}
    if (REDUCE || document.hidden || !HAS3D()) { doReveal(); return; }
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

    const cv = root.querySelector("canvas");
    const R = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
    R.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x05070d, 9, 26);
    const cam = new THREE.PerspectiveCamera(38, 1, 0.1, 100);

    // floor grid, receding into fog
    const grid = new THREE.GridHelper(60, 60, 0x2a3f6e, 0x142038);
    grid.position.y = -0.01; scene.add(grid);
    scene.add(new THREE.AmbientLight(0x8fa3c8, 0.55));
    const dl = new THREE.DirectionalLight(0xffffff, 0.9); dl.position.set(4, 9, 6); scene.add(dl);

    // the price path: bottom-left to top-right, with realistic wiggles
    const N = 90, pts = [];
    for (let i = 0; i <= N; i++) {
      const u = i / N, x = -6.2 + 12.4 * u;
      const y = 0.35 + 4.9 * Math.pow(u, 1.25) + 0.28 * Math.sin(u * 23) * (1 - u * 0.4) + 0.16 * Math.sin(u * 57 + 1.3) - 0.35 * Math.exp(-((u - 0.55) ** 2) / 0.004);
      pts.push(new THREE.Vector3(x, Math.max(0.15, y), 0));
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const SEG = 360, RAD = 10;
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, SEG, 0.045, RAD, false), new THREE.MeshBasicMaterial({ color: 0xbfdcff }));
    const glow = new THREE.Mesh(new THREE.TubeGeometry(curve, SEG, 0.16, RAD, false),
      new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(tube, glow);

    // translucent curtain under the line (alpha by height)
    const cpos = [], ch = [];
    const sp = curve.getSpacedPoints(SEG);
    for (let i = 0; i < SEG; i++) {
      const a = sp[i], b = sp[i + 1];
      cpos.push(a.x, 0, 0, b.x, 0, 0, b.x, b.y, 0, a.x, 0, 0, b.x, b.y, 0, a.x, a.y, 0);
      ch.push(0, 0, 1, 0, 1, 1);
    }
    const cg = new THREE.BufferGeometry();
    cg.setAttribute("position", new THREE.Float32BufferAttribute(cpos, 3));
    cg.setAttribute("aH", new THREE.Float32BufferAttribute(ch, 1));
    const curtain = new THREE.Mesh(cg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      vertexShader: "attribute float aH; varying float vH; void main(){ vH=aH; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
      fragmentShader: "varying float vH; void main(){ gl_FragColor=vec4(0.23,0.51,0.96, 0.05+0.30*vH*vH); }",
    }));
    scene.add(curtain);

    // volume bars rising behind the line
    const bars = [], NB = 34;
    for (let i = 0; i < NB; i++) {
      const u = (i + 0.5) / NB, h = 0.25 + 1.6 * Math.abs(Math.sin(i * 1.7) * 0.6 + Math.sin(i * 0.37) * 0.5) * (0.5 + u);
      const up = Math.sin(i * 2.3) > -0.35;
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1, 0.2), new THREE.MeshStandardMaterial({
        color: up ? 0x1f4fd6 : 0x8a2f3a, emissive: up ? 0x0b2a7a : 0x3a0d12, roughness: 0.4, metalness: 0.2 }));
      m.position.set(-6.2 + 12.4 * u, 0, -1.4); m.userData = { h, u }; m.scale.y = 0.001; scene.add(m); bars.push(m);
    }

    // arrowhead riding the tip, with its own light
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.55, 28), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x7cb0ff, emissiveIntensity: 0.9 }));
    const tipLight = new THREE.PointLight(0x7cb0ff, 2.4, 7);
    scene.add(arrow, tipLight);

    // sparks drifting up
    const NP = 420, pp = new Float32Array(NP * 3);
    for (let i = 0; i < NP; i++) { pp[i * 3] = (Math.random() - 0.5) * 22; pp[i * 3 + 1] = Math.random() * 9; pp[i * 3 + 2] = (Math.random() - 0.5) * 12 - 2; }
    const pg = new THREE.BufferGeometry(); pg.setAttribute("position", new THREE.BufferAttribute(pp, 3));
    const sparks = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x9fc7ff, size: 0.045, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending }));
    scene.add(sparks);

    // burn pass: scene -> texture -> full-screen quad that dissolves from the centre with an ember edge
    let rt = null;
    const post = new THREE.Scene(), ocam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const U = { tScene: { value: null }, uProg: { value: -1 }, uTime: { value: 0 }, uAspect: { value: 1 } };
    post.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      uniforms: U, transparent: true, depthTest: false,
      vertexShader: "varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.0,1.0); }",
      fragmentShader: `uniform sampler2D tScene; uniform float uProg, uTime, uAspect; varying vec2 vUv;
        float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
        float n(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
          return mix(mix(h(i),h(i+vec2(1,0)),f.x), mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x), f.y); }
        float fbm(vec2 p){ float v=0.0, a=0.5; for(int k=0;k<5;k++){ v+=a*n(p); p*=2.03; a*=0.5; } return v; }
        void main(){
          vec4 c = texture2D(tScene, vUv);
          vec2 p = vUv-0.5; p.x *= uAspect;
          float field = length(p)*1.1 + (fbm(vUv*vec2(uAspect,1.0)*3.5 + uTime*0.05)-0.5)*0.55;
          float e = field - uProg;
          if (e < 0.0) discard;
          float glow = 1.0 - smoothstep(0.0, 0.035, e);
          float charr = 1.0 - smoothstep(0.0, 0.16, e);
          vec3 ember = mix(vec3(1.0,0.28,0.04), vec3(1.0,0.86,0.55), glow*glow);
          vec3 col = mix(c.rgb, c.rgb*0.18 + vec3(0.06,0.02,0.0), charr*0.85);
          col += ember * glow * 2.2 + vec3(1.0,0.45,0.1) * pow(charr, 3.0) * 0.35 * step(-0.5, uProg);
          gl_FragColor = vec4(col, 1.0);
        }`,
    })));

    function size() {
      const w = innerWidth, hh = innerHeight;
      R.setSize(w, hh, false);
      cam.aspect = w / hh; cam.updateProjectionMatrix();
      const pr = R.getPixelRatio();
      if (rt) rt.dispose();
      rt = new THREE.WebGLRenderTarget(Math.floor(w * pr), Math.floor(hh * pr));
      U.tScene.value = rt.texture; U.uAspect.value = w / hh;
    }
    size(); addEventListener("resize", size);

    const t0 = performance.now(), MIN = short ? 0 : 3000;
    setTimeout(() => { readyFlag = true; }, short ? 0 : 4500);       // never wait forever for data
    let prog = short ? 1 : 0, burnT = null, done = false, lastStep = -1;
    const pct = root.querySelector("#fx-pct"), bar = root.querySelector("#fx-barfill"), status = root.querySelector("#fx-status");
    const skip = () => { prog = 1; readyFlag = true; if (burnT == null) startBurn(); };
    root.querySelector("#fx-skip").onclick = skip;
    addEventListener("keydown", function k(e) { if (e.key === "Escape") { skip(); removeEventListener("keydown", k); } });
    function startBurn() { burnT = performance.now(); root.classList.add("burning"); document.documentElement.style.overflow = ""; doReveal(); }

    const tmp = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    function frame(now) {
      if (done) return;
      const t = (now - t0) / 1000;
      if (burnT == null) {
        const target = Math.min(1, (now - t0) / Math.max(1, MIN)) * (readyFlag ? 1 : 0.92);
        prog += (target - prog) * (short ? 1 : 0.08);
        if (prog > 0.995 && readyFlag && now - t0 > MIN) { prog = 1; startBurn(); }
      }
      const p = Math.min(1, prog);
      // grow the line, curtain, bars; move the arrow
      const seg = Math.max(1, Math.floor(p * SEG));
      tube.geometry.setDrawRange(0, seg * RAD * 6); glow.geometry.setDrawRange(0, seg * RAD * 6); cg.setDrawRange(0, seg * 6);
      const u = Math.max(0.002, Math.min(0.999, p));
      const tip = curve.getPointAt(u), tan = curve.getTangentAt(u);
      arrow.position.copy(tip).addScaledVector(tan, 0.18); arrow.quaternion.setFromUnitVectors(up, tan);
      tipLight.position.copy(tip).add(tmp.set(0, 0.4, 0.8));
      bars.forEach(b => { const k = Math.max(0, Math.min(1, (p - b.userData.u * 0.95) * 6)); b.visible = k > 0; b.scale.y = 0.001 + b.userData.h * k; b.position.y = b.scale.y / 2; });
      const pos = pg.attributes.position.array;
      for (let i = 0; i < NP; i++) { pos[i * 3 + 1] += 0.006 + (i % 7) * 0.001; if (pos[i * 3 + 1] > 9) pos[i * 3 + 1] = 0; }
      pg.attributes.position.needsUpdate = true;
      // camera: slow dolly towards the tip, gentle sway
      const narrow = cam.aspect < 1 ? 2.1 / Math.max(0.45, cam.aspect) * 0.62 : Math.max(1, 1.45 / cam.aspect);
      cam.position.set(-1.6 + 1.4 * p + Math.sin(t * 0.4) * 0.2, 3.0 + 0.6 * p, (14.2 - 1.4 * p) * narrow);
      cam.lookAt(0.5, 2.55, 0);
      // UI
      if (pct) pct.textContent = Math.round(p * 100);
      if (bar) bar.style.width = (p * 100).toFixed(1) + "%";
      const st = Math.min(STEPS.length - 1, Math.floor(p * STEPS.length));
      if (status && st !== lastStep) { lastStep = st; status.innerHTML = STEPS.map((s, i) => i < st ? `${s} ✓` : i === st ? `<b>${s}</b>` : "").filter(Boolean).slice(-3).join("<br>"); }
      // burn
      if (burnT != null) {
        const b = (now - burnT) / (short ? 750 : 1300);
        U.uProg.value = -0.15 + b * 1.25;
        if (b >= 1) { done = true; R.dispose(); rt.dispose(); root.remove(); removeEventListener("resize", size); return; }
      }
      U.uTime.value = t;
      R.setRenderTarget(rt); R.setClearColor(0x05070d, 1); R.clear(); R.render(scene, cam);
      R.setRenderTarget(null); R.setClearColor(0x000000, 0); R.clear(); R.render(post, ocam);
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

  // shared bits for the two data stages
  function stage(el, camPos) {
    const cv = document.createElement("canvas"); el.prepend(cv);
    const R = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
    R.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
    scene.add(new THREE.AmbientLight(0x9fb3d8, 0.6));
    const d = new THREE.DirectionalLight(0xffffff, 0.95); d.position.set(6, 12, 8); scene.add(d);
    const d2 = new THREE.DirectionalLight(0x7cb0ff, 0.35); d2.position.set(-8, 4, -6); scene.add(d2);
    let yaw = 0.62, pitch = 0.5, dist = camPos, drag = null, idle = 0, visible = true, raf = 0;
    cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", e => { if (!drag) return; yaw = drag[2] - (e.clientX - drag[0]) * 0.006; pitch = Math.max(0.12, Math.min(1.25, drag[3] + (e.clientY - drag[1]) * 0.004)); idle = 0; });
    cv.addEventListener("pointerup", () => { drag = null; });
    cv.addEventListener("wheel", e => { e.preventDefault(); dist = Math.max(camPos * 0.55, Math.min(camPos * 1.6, dist * (1 + Math.sign(e.deltaY) * 0.08))); }, { passive: false });
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return; R.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && !raf) loop(); }).observe(el);
    const api = { scene, cam, R, target: new THREE.Vector3(0, 0, 0), onFrame: null };
    function loop() {
      raf = 0; if (!visible || document.hidden) return;
      if (!drag && !REDUCE) { idle += 1; yaw += 0.0012; }
      const dd = dist * (cam.aspect < 1 ? 1.35 : 1);
      cam.position.set(api.target.x + dd * Math.cos(pitch) * Math.sin(yaw), api.target.y + dd * Math.sin(pitch), api.target.z + dd * Math.cos(pitch) * Math.cos(yaw));
      cam.lookAt(api.target);
      if (api.onFrame) api.onFrame();
      R.render(scene, cam);
      raf = requestAnimationFrame(loop);
    }
    document.addEventListener("visibilitychange", () => { if (!document.hidden && visible && !raf) loop(); });
    loop();
    return api;
  }
  function noStage(el, why) { const f = document.createElement("div"); f.className = "fx-fallback"; f.textContent = why; el.appendChild(f); return { update() {}, set() {} }; }

  // ------------------------------------------------------------------------------------------------ live 3D liquidity terrain
  FX.terrain = function (el) {
    if (!HAS3D()) return noStage(el, "3D view needs WebGL");
    const S = stage(el, 30);
    const W = 41, T = 48, GAP = 0.32, DEP = 0.36;            // price levels (+-20 ticks), time columns
    const geo = new THREE.BoxGeometry(GAP * 0.86, 1, DEP * 0.8); geo.translate(0, 0.5, 0);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.1 });
    const mesh = new THREE.InstancedMesh(geo, mat, W * T);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    S.scene.add(mesh);
    const floor = new THREE.GridHelper(40, 40, 0x24365e, 0x121c33); floor.position.y = -0.02; S.scene.add(floor);
    const midPlane = new THREE.Mesh(new THREE.PlaneGeometry(0.06, T * DEP + 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
    midPlane.rotation.x = -Math.PI / 2; midPlane.position.y = 0.01; S.scene.add(midPlane);
    S.target.set(0, 0.6, 0);
    const m4 = new THREE.Matrix4(), col = new THREE.Color(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), ps = new THREE.Vector3();
    const BID = [new THREE.Color(0x0b3d2e), new THREE.Color(0x16c784), new THREE.Color(0xd9fff0)];
    const ASK = [new THREE.Color(0x3d0b12), new THREE.Color(0xea3943), new THREE.Color(0xffe0e3)];
    const ramp = (r, k) => k < 0.5 ? col.copy(r[0]).lerp(r[1], k * 2) : col.copy(r[1]).lerp(r[2], (k - 0.5) * 2);
    let target = new Float32Array(W * T), cur = new Float32Array(W * T), side = new Int8Array(W * T), kk = new Float32Array(W * T);
    S.onFrame = () => {
      for (let i = 0; i < W * T; i++) {
        cur[i] += (target[i] - cur[i]) * 0.12;
        const ti = Math.floor(i / W), pi = i % W;
        ps.set((pi - (W - 1) / 2) * GAP, 0, (ti - (T - 1) / 2) * DEP);
        sc.set(1, Math.max(0.001, cur[i]), 1);
        m4.compose(ps, q, sc); mesh.setMatrixAt(i, m4);
        mesh.setColorAt(i, cur[i] < 0.002 ? col.set(0x0a1020) : ramp(side[i] > 0 ? BID : ASK, kk[i]));
      }
      mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    };
    return {
      /* columns: [ts, lowestTick, bidTick, askTick, sizes[]] oldest first (the page's heatmap history) */
      update(cols) {
        if (!cols || !cols.length) return;
        const last = cols[cols.length - 1], mid = Math.round((last[2] + last[3]) / 2);
        const use = cols.slice(-T), vals = [];
        use.forEach(c => c[4].forEach(v => v > 0 && vals.push(v)));
        vals.sort((a, b) => a - b);
        const cap = vals.length ? vals[Math.floor(0.97 * (vals.length - 1))] : 1;
        target.fill(0); kk.fill(0);
        use.forEach((c, j) => {
          const ti = T - use.length + j, cm = Math.round((c[2] + c[3]) / 2);
          for (let pi = 0; pi < W; pi++) {
            const tk = mid - (W - 1) / 2 + pi, v = c[4][tk - c[1]] || 0, i = ti * W + pi;
            const k = Math.min(1, Math.sqrt(v / cap));
            target[i] = k * 3.2; kk[i] = k; side[i] = tk <= c[2] ? 1 : -1;
          }
        });
      },
    };
  };

  // ------------------------------------------------------------------------------------------------ 3D bars (results)
  FX.bars3d = function (el) {
    if (!HAS3D()) return noStage(el, "3D view needs WebGL");
    const S = stage(el, 19);
    const group = new THREE.Group(); S.scene.add(group);
    const floor = new THREE.GridHelper(40, 40, 0x24365e, 0x121c33); S.scene.add(floor);
    const zero = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshBasicMaterial({ color: 0x0b1428, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    zero.rotation.x = -Math.PI / 2; S.scene.add(zero);
    let items = [];
    S.onFrame = () => items.forEach(b => {
      b.cur += (b.to - b.cur) * 0.07;
      const h = Math.max(0.001, Math.abs(b.cur));
      b.mesh.scale.y = h; b.mesh.position.y = b.cur >= 0 ? h / 2 : -h / 2;
    });
    return {
      set(cats, series, colors) {   // series: {name: [values per category]}
        group.clear(); items = [];
        const names = Object.keys(series), all = names.flatMap(n => series[n]).filter(v => v != null).map(Math.abs);
        const mx = Math.max(1e-9, ...all), H = 5.5, gx = 1.25, gz = 1.1;
        cats.forEach((c, i) => names.forEach((n, j) => {
          const v = series[n][i]; if (v == null) return;
          const m = new THREE.Mesh(new THREE.BoxGeometry(0.62, 1, 0.62), new THREE.MeshStandardMaterial({
            color: colors[n], emissive: new THREE.Color(colors[n]).multiplyScalar(0.25), roughness: 0.35, metalness: 0.15, transparent: v < 0, opacity: v < 0 ? 0.78 : 1 }));
          m.position.set((i - (cats.length - 1) / 2) * gx, 0, (j - (names.length - 1) / 2) * gz);
          group.add(m); items.push({ mesh: m, to: (v / mx) * H, cur: 0 });
        }));
        S.target.set(0, 0.8, 0);
      },
    };
  };

  window.FX = FX;
})();
