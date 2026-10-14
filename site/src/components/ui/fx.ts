// Celebration effects: a particle burst drawn on a temporary full-screen canvas.

const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const PALETTE = ["#4fd1ff", "#b388ff", "#ff5ca8", "#3ddc97", "#ffb547"];

interface Particle {
  x: number; y: number; vx: number; vy: number; life: number; size: number; color: string; spin: number; angle: number; shape: 0 | 1;
}

/**
 * Burst of particles from a viewport point.
 * `count` and `power` scale it from a small sparkle (fill-in) to full confetti (all tests pass).
 */
export function burst(x: number, y: number, { count = 40, power = 1, gravity = 0.18, spread = Math.PI * 2 } = {}) {
  if (typeof window === "undefined" || reducedMotion()) return;
  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  Object.assign(canvas.style, { position: "fixed", inset: "0", width: "100vw", height: "100vh", pointerEvents: "none", zIndex: "9999" });
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);

  const parts: Particle[] = Array.from({ length: count }, () => {
    const a = -Math.PI / 2 + (Math.random() - 0.5) * spread;
    const v = (2 + Math.random() * 5) * power;
    return {
      x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, size: 3 + Math.random() * 4,
      color: PALETTE[(Math.random() * PALETTE.length) | 0], spin: (Math.random() - 0.5) * 0.4, angle: Math.random() * 6, shape: Math.random() < 0.5 ? 0 : 1,
    };
  });

  const tick = () => {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    let alive = 0;
    for (const p of parts) {
      if (p.life <= 0) continue;
      alive++;
      p.vy += gravity;
      p.vx *= 0.985;
      p.x += p.vx;
      p.y += p.vy;
      p.angle += p.spin;
      p.life -= 0.012;
      ctx.save();
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.angle);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      if (p.shape === 0) ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      else { ctx.beginPath(); ctx.arc(0, 0, p.size / 2.5, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    }
    if (alive) requestAnimationFrame(tick);
    else canvas.remove();
  };
  requestAnimationFrame(tick);
}

export function burstFrom(el: Element | null, opts?: Parameters<typeof burst>[2]) {
  if (!el) return;
  const r = el.getBoundingClientRect();
  burst(r.left + r.width / 2, r.top + r.height / 2, opts);
}
