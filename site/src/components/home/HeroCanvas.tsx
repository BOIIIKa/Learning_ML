import { useEffect, useRef } from "react";
import { contours } from "d3-contour";

/**
 * Background animation for the home hero: particles running (noisy) gradient descent on a
 * multi-modal loss surface drawn as blueprint contour lines — SGD, literally.
 * Colours come from the theme's ink/accent tokens, so it follows the light/dark switch.
 */
export default function HeroCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let frame = 0;

    // Loss: a sum of Gaussian wells. Gradient computed analytically.
    const wells = [
      { x: 0.25, y: 0.35, s: 0.12, d: 1.0 },
      { x: 0.7, y: 0.6, s: 0.16, d: 1.3 },
      { x: 0.55, y: 0.2, s: 0.09, d: 0.7 },
      { x: 0.85, y: 0.25, s: 0.1, d: 0.8 },
      { x: 0.15, y: 0.8, s: 0.13, d: 0.9 },
    ];
    const grad = (x: number, y: number) => {
      let gx = 0;
      let gy = 0;
      for (const k of wells) {
        const e = k.d * Math.exp(-((x - k.x) ** 2 + (y - k.y) ** 2) / (2 * k.s * k.s));
        gx += (e * (x - k.x)) / (k.s * k.s);
        gy += (e * (y - k.y)) / (k.s * k.s);
      }
      return [gx, gy];
    };

    const loss = (x: number, y: number) =>
      -wells.reduce((acc, k) => acc + k.d * Math.exp(-((x - k.x) ** 2 + (y - k.y) ** 2) / (2 * k.s * k.s)), 0);

    // Particles: mostly ink, one in eight in the highlighter accent.
    const spawn = () => ({ x: Math.random(), y: Math.random(), age: 0, hot: Math.random() < 0.125 });
    const parts = Array.from({ length: 110 }, spawn);

    // Trails live on an offscreen buffer; each frame = contours + trails.
    const trail = document.createElement("canvas");
    const tctx = trail.getContext("2d")!;
    const contourLayer = document.createElement("canvas");
    const cctx = contourLayer.getContext("2d")!;
    let ink = "#eaf3ff";
    let accent = "#ffcc4d";
    const readTheme = () => {
      const cs = getComputedStyle(document.documentElement);
      ink = cs.getPropertyValue("--ink").trim() || ink;
      accent = cs.getPropertyValue("--accent").trim() || accent;
    };

    const drawContours = () => {
      const gw = 96;
      const gh = Math.max(24, Math.round((gw * h) / Math.max(w, 1)));
      const vals = new Array(gw * gh);
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) vals[j * gw + i] = loss(i / (gw - 1), j / (gh - 1));
      const levels = contours().size([gw, gh]).thresholds([-1.2, -1.0, -0.8, -0.6, -0.45, -0.3, -0.18, -0.08])(vals);
      cctx.clearRect(0, 0, w, h);
      cctx.strokeStyle = ink;
      cctx.lineWidth = 1;
      levels.forEach((lv, li) => {
        cctx.globalAlpha = 0.1 + 0.03 * li;
        cctx.setLineDash(li % 2 ? [4, 5] : []);
        cctx.beginPath();
        for (const poly of lv.coordinates)
          for (const ring of poly)
            ring.forEach(([cx, cy], k) => {
              const X = (cx / (gw - 1)) * w;
              const Y = (cy / (gh - 1)) * h;
              if (k === 0) cctx.moveTo(X, Y);
              else cctx.lineTo(X, Y);
            });
        cctx.stroke();
      });
      cctx.globalAlpha = 1;
      cctx.setLineDash([]);
      // Minimum markers: small crosses at each well, like datum points on a drawing.
      cctx.strokeStyle = accent;
      cctx.globalAlpha = 0.7;
      for (const k of wells) {
        const X = k.x * w;
        const Y = k.y * h;
        cctx.beginPath();
        cctx.moveTo(X - 6, Y); cctx.lineTo(X + 6, Y);
        cctx.moveTo(X, Y - 6); cctx.lineTo(X, Y + 6);
        cctx.stroke();
      }
      cctx.globalAlpha = 1;
    };

    const resize = () => {
      const dpr = devicePixelRatio || 1;
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      for (const c of [canvas, trail, contourLayer]) {
        c.width = w * dpr;
        c.height = h * dpr;
      }
      for (const c of [ctx, tctx, cctx]) c.setTransform(dpr, 0, 0, dpr, 0, 0);
      readTheme();
      drawContours();
    };
    resize();
    addEventListener("resize", resize);
    // Re-ink when the theme toggles.
    const mo = new MutationObserver(() => { readTheme(); drawContours(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    const tick = () => {
      // Fade the previous frame to leave trails.
      tctx.globalCompositeOperation = "destination-out";
      tctx.fillStyle = "rgba(0,0,0,0.05)";
      tctx.fillRect(0, 0, w, h);
      tctx.globalCompositeOperation = "source-over";
      for (const p of parts) {
        const [gx, gy] = grad(p.x, p.y);
        const nx = p.x - 0.0022 * gx + (Math.random() - 0.5) * 0.004;
        const ny = p.y - 0.0022 * gy + (Math.random() - 0.5) * 0.004;
        tctx.strokeStyle = p.hot ? accent : ink;
        tctx.globalAlpha = p.hot ? 0.85 : 0.45;
        tctx.lineWidth = p.hot ? 1.6 : 1.1;
        tctx.beginPath();
        tctx.moveTo(p.x * w, p.y * h);
        tctx.lineTo(nx * w, ny * h);
        tctx.stroke();
        p.x = nx;
        p.y = ny;
        p.age++;
        if (p.age > 260 + Math.random() * 200 || p.x < -0.05 || p.x > 1.05 || p.y < -0.05 || p.y > 1.05) Object.assign(p, spawn());
      }
      tctx.globalAlpha = 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(contourLayer, 0, 0);
      ctx.drawImage(trail, 0, 0);
      frame = requestAnimationFrame(tick);
    };
    if (!reduced) frame = requestAnimationFrame(tick);
    else ctx.drawImage(contourLayer, 0, 0, w, h);
    return () => {
      mo.disconnect();
      cancelAnimationFrame(frame);
      removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={ref} className="hero-canvas" aria-hidden="true" />;
}
