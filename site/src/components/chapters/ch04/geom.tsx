// Small SVG helpers shared by the chapter 04 widgets.

export type Box = { x: [number, number]; y: [number, number] };
type P = [number, number];

/** The part of the box where w·p + c ≥ 0 (one Sutherland–Hodgman clipping pass). */
export function halfPlane(w: P, c: number, box: Box): P[] {
  const corners: P[] = [[box.x[0], box.y[0]], [box.x[1], box.y[0]], [box.x[1], box.y[1]], [box.x[0], box.y[1]]];
  const f = (p: P) => w[0] * p[0] + w[1] * p[1] + c;
  const out: P[] = [];
  for (let k = 0; k < 4; k++) {
    const a = corners[k];
    const b = corners[(k + 1) % 4];
    const fa = f(a);
    const fb = f(b);
    if (fa >= 0) out.push(a);
    if ((fa >= 0) !== (fb >= 0)) {
      const t = fa / (fa - fb);
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
  }
  return out;
}

/** Endpoints of the line w·p + c = 0 inside the box, or null. */
export function lineInBox(w: P, c: number, box: Box): [P, P] | null {
  const pts: P[] = [];
  if (Math.abs(w[1]) > 1e-12) {
    for (const x of box.x) {
      const y = -(w[0] * x + c) / w[1];
      if (y >= box.y[0] - 1e-9 && y <= box.y[1] + 1e-9) pts.push([x, y]);
    }
  }
  if (Math.abs(w[0]) > 1e-12) {
    for (const y of box.y) {
      const x = -(w[1] * y + c) / w[0];
      if (x >= box.x[0] - 1e-9 && x <= box.x[1] + 1e-9) pts.push([x, y]);
    }
  }
  return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : null;
}

export const f1 = (v: number) => v.toFixed(1);

/** A straight arrow in screen coordinates with a filled triangular head. */
export function Arrow({ from, to, cls, head = 9 }: { from: P; to: P; cls: string; head?: number }) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (len < 1) return null;
  const ux = dx / len;
  const uy = dy / len;
  const h = Math.min(head, len * 0.6);
  const base: P = [to[0] - ux * h, to[1] - uy * h];
  const left: P = [base[0] - uy * h * 0.5, base[1] + ux * h * 0.5];
  const right: P = [base[0] + uy * h * 0.5, base[1] - ux * h * 0.5];
  return (
    <g>
      <line x1={f1(from[0])} y1={f1(from[1])} x2={f1(base[0])} y2={f1(base[1])} className={`lc-arrow ${cls}`} />
      <polygon points={[to, left, right].map((p) => `${f1(p[0])},${f1(p[1])}`).join(" ")} className={`lc-arrow-head ${cls}`} />
    </g>
  );
}
