import { useEffect, useRef } from "react";

/**
 * Background animation for the home hero: particles running (noisy) gradient descent on a
 * multi-modal loss surface, leaving glowing trails — SGD, literally.
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

    const colors = ["79,209,255", "179,136,255", "255,92,168"];
    const spawn = () => ({ x: Math.random(), y: Math.random(), age: 0, c: colors[(Math.random() * 3) | 0] });
    const parts = Array.from({ length: 140 }, spawn);

    const resize = () => {
      const dpr = devicePixelRatio || 1;
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    addEventListener("resize", resize);

    const tick = () => {
      // Fade the previous frame to leave trails.
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,0.06)";
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      for (const p of parts) {
        const [gx, gy] = grad(p.x, p.y);
        const nx = p.x - 0.0022 * gx + (Math.random() - 0.5) * 0.004;
        const ny = p.y - 0.0022 * gy + (Math.random() - 0.5) * 0.004;
        ctx.strokeStyle = `rgba(${p.c},0.55)`;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(p.x * w, p.y * h);
        ctx.lineTo(nx * w, ny * h);
        ctx.stroke();
        p.x = nx;
        p.y = ny;
        p.age++;
        if (p.age > 260 + Math.random() * 200 || p.x < -0.05 || p.x > 1.05 || p.y < -0.05 || p.y > 1.05) Object.assign(p, spawn());
      }
      frame = requestAnimationFrame(tick);
    };
    if (!reduced) frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={ref} className="hero-canvas" aria-hidden="true" />;
}
