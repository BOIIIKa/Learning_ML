// Linear regression and regularization for the chapter 03 visuals:
// projection onto a column space, ridge / lasso (closed form, soft thresholding, coordinate descent),
// constrained quadratics on the L1 / L2 ball, polynomial regularization paths with cross-validation.
import { gaussian, r6, rng } from "./data";

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

// ---------------------------------------------------------------- small helpers

export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

/** Solve A w = b by Gaussian elimination with partial pivoting (small dense systems). */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const w = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * w[k];
    w[r] = s / M[r][r];
  }
  return w;
}

/** Soft-thresholding operator S_λ(z) = sign(z) · max(|z| − λ, 0). */
export const softThreshold = (z: number, lam: number) => Math.sign(z) * Math.max(Math.abs(z) - lam, 0);

// ---------------------------------------------------------------- least squares as projection

/**
 * Least squares for a design matrix X (n × p, given as columns) and target y:
 * coefficients, the projection ŷ = Xw, the residual e = y − ŷ and Xᵀe (≈ 0).
 */
export function projectOntoColumns(cols: number[][], y: number[]) {
  const p = cols.length;
  const G = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => dot(cols[i], cols[j])));
  const w = solveLinear(G, cols.map((c) => dot(c, y)));
  const yhat = y.map((_, k) => cols.reduce((s, c, j) => s + w[j] * c[k], 0));
  const e = y.map((v, k) => v - yhat[k]);
  return { w, yhat, e, xte: cols.map((c) => dot(c, e)), rss: dot(e, e) };
}

/** Diagonal of the hat matrix H = X (XᵀX)⁻¹ Xᵀ (the leverages). */
export function leverages(cols: number[][]): number[] {
  const n = cols[0].length;
  return Array.from({ length: n }, (_, i) => {
    const ei = Array.from({ length: n }, (_, k) => (k === i ? 1 : 0));
    return projectOntoColumns(cols, ei).yhat[i];
  });
}

/**
 * An orthographic camera looking at the origin: azimuth θ around the "vertical" axis, elevation φ.
 * `basis` = (e1, e2, vertical) is an orthonormal frame, by default the world axes with z up.
 */
export function camera(theta: number, phi: number, basis: [Vec3, Vec3, Vec3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
  const [b1, b2, b3] = basis;
  const comb = (a: number, b: number, c: number): Vec3 => [0, 1, 2].map((i) => a * b1[i] + b * b2[i] + c * b3[i]) as Vec3;
  const right = comb(-Math.sin(theta), Math.cos(theta), 0);
  const up = comb(-Math.sin(phi) * Math.cos(theta), -Math.sin(phi) * Math.sin(theta), Math.cos(phi));
  const toward = comb(Math.cos(phi) * Math.cos(theta), Math.cos(phi) * Math.sin(theta), Math.sin(phi));
  return {
    right,
    up,
    /** Screen coordinates (right, up) and depth (larger = nearer the viewer). */
    project: (p: number[]): [number, number, number] => [dot(p, right), dot(p, up), dot(p, toward)],
    /** A screen-space displacement mapped back to a 3-D displacement parallel to the screen. */
    unproject: (dx: number, dy: number): Vec3 => [dx * right[0] + dy * up[0], dx * right[1] + dy * up[1], dx * right[2] + dy * up[2]],
  };
}

// ---------------------------------------------------------------- the L1 / L2 ball

/**
 * The quadratic loss ½ (w − c)ᵀ A (w − c), whose contours are ellipses centred at the
 * unconstrained optimum c. `corr` ∈ (−1, 1) tilts the ellipses, `ratio` stretches them.
 */
export function quadForm(corr: number, ratio = 1): [[number, number], [number, number]] {
  return [[1, corr * Math.sqrt(ratio)], [corr * Math.sqrt(ratio), ratio]];
}

export const quadLoss = (A: number[][], c: Vec2, w: Vec2) => {
  const d = [w[0] - c[0], w[1] - c[1]];
  return 0.5 * (A[0][0] * d[0] * d[0] + 2 * A[0][1] * d[0] * d[1] + A[1][1] * d[1] * d[1]);
};

/** argmin ½(w − c)ᵀA(w − c) subject to ‖w‖₂ ≤ t, via w(μ) = (A + μI)⁻¹ A c and bisection on μ. */
export function solveL2Ball(A: number[][], c: Vec2, t: number): { w: Vec2; mu: number } {
  if (Math.hypot(c[0], c[1]) <= t) return { w: [c[0], c[1]], mu: 0 };
  const Ac = [A[0][0] * c[0] + A[0][1] * c[1], A[1][0] * c[0] + A[1][1] * c[1]];
  const at = (mu: number): Vec2 => {
    const a = A[0][0] + mu, b = A[0][1], d = A[1][1] + mu;
    const det = a * d - b * b;
    return [(d * Ac[0] - b * Ac[1]) / det, (a * Ac[1] - b * Ac[0]) / det];
  };
  let lo = 0, hi = 1;
  while (Math.hypot(...at(hi)) > t) hi *= 2;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (Math.hypot(...at(mid)) > t) lo = mid;
    else hi = mid;
  }
  return { w: at(hi), mu: hi };
}

/** argmin ½(w − c)ᵀA(w − c) subject to ‖w‖₁ ≤ t: check the interior, then each edge of the diamond. */
export function solveL1Ball(A: number[][], c: Vec2, t: number): { w: Vec2; vertex: boolean } {
  if (Math.abs(c[0]) + Math.abs(c[1]) <= t) return { w: [c[0], c[1]], vertex: false };
  const verts: Vec2[] = [[t, 0], [0, t], [-t, 0], [0, -t]];
  let best: Vec2 = verts[0];
  let bestF = Infinity;
  for (let k = 0; k < 4; k++) {
    const p = verts[k], q = verts[(k + 1) % 4];
    const d: Vec2 = [q[0] - p[0], q[1] - p[1]];
    // Minimize the 1-D quadratic along p + s d, s ∈ [0, 1].
    const Ad = [A[0][0] * d[0] + A[0][1] * d[1], A[1][0] * d[0] + A[1][1] * d[1]];
    const curv = d[0] * Ad[0] + d[1] * Ad[1];
    const g = (p[0] - c[0]) * Ad[0] + (p[1] - c[1]) * Ad[1];
    const s = Math.max(0, Math.min(1, -g / curv));
    const w: Vec2 = [p[0] + s * d[0], p[1] + s * d[1]];
    const f = quadLoss(A, c, w);
    if (f < bestF - 1e-12) { bestF = f; best = w; }
  }
  const vertex = Math.abs(best[0]) < 1e-9 * Math.max(1, t) || Math.abs(best[1]) < 1e-9 * Math.max(1, t);
  return { w: best, vertex };
}

/** Coordinate descent on ½(w − c)ᵀA(w − c) + λ‖w‖₁ in 2-D, recording every half-step. */
export function lassoCD2(A: number[][], c: Vec2, lam: number, sweeps = 12): Vec2[] {
  const Ac = [A[0][0] * c[0] + A[0][1] * c[1], A[1][0] * c[0] + A[1][1] * c[1]];
  const w: Vec2 = [0, 0];
  const path: Vec2[] = [[0, 0]];
  for (let s = 0; s < sweeps; s++) {
    for (let j = 0; j < 2; j++) {
      const k = 1 - j;
      w[j] = softThreshold(Ac[j] - A[j][k] * w[k], lam) / A[j][j];
      path.push([w[0], w[1]]);
    }
  }
  return path;
}

// ---------------------------------------------------------------- polynomial regularization paths

/** Fuel use (L/100 km) at different speeds (km/h): a U-shaped truth plus noise. */
export const fuelTruth = (v: number) => 4 + 0.0009 * (v - 60) ** 2;

export function fuelData(n = 12, seed = 1, sigma = 0.4) {
  const rand = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = r6(20 + (100 * (i + rand())) / n);
    xs.push(v);
    ys.push(r6(fuelTruth(v) + sigma * gaussian(rand)));
  }
  return { xs, ys };
}

export const DEGREE = 9;
const polyRow = (v: number, deg: number) => {
  const u = (v - 70) / 50;
  return Array.from({ length: deg }, (_, k) => u ** (k + 1));
};

/** Polynomial features u, u², …, u^deg, each standardized on the training set; the intercept is ȳ. */
export function polyDesign(xs: number[], deg = DEGREE) {
  const F = xs.map((v) => polyRow(v, deg));
  const n = xs.length;
  const mu = Array.from({ length: deg }, (_, j) => F.reduce((s, r) => s + r[j], 0) / n);
  const sd = Array.from({ length: deg }, (_, j) => Math.sqrt(F.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n));
  const Z = F.map((r) => r.map((v, j) => (v - mu[j]) / sd[j]));
  const transform = (v: number) => polyRow(v, deg).map((f, j) => (f - mu[j]) / sd[j]);
  return { Z, transform };
}

/** Ridge: argmin (1/2n)‖y − Zw‖² + (λ/2)‖w‖², i.e. (ZᵀZ/n + λI) w = Zᵀy/n. */
export function ridgeFit(Z: number[][], y: number[], lam: number): number[] {
  const n = Z.length, p = Z[0].length;
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Z.reduce((s, r) => s + r[i] * r[j], 0) / n + (i === j ? lam : 0)));
  return solveLinear(A, Array.from({ length: p }, (_, j) => Z.reduce((s, r, k) => s + r[j] * y[k], 0) / n));
}

/** Lasso: argmin (1/2n)‖y − Zw‖² + λ‖w‖₁ by cyclic coordinate descent with soft thresholding. */
export function lassoFit(Z: number[][], y: number[], lam: number, w0?: number[], maxSweeps = 3000, tol = 1e-8) {
  const n = Z.length, p = Z[0].length;
  const w = w0 ? [...w0] : new Array(p).fill(0);
  const r = y.map((v, k) => v - dot(Z[k], w));
  const sq = Array.from({ length: p }, (_, j) => Z.reduce((s, row) => s + row[j] * row[j], 0) / n);
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let change = 0;
    for (let j = 0; j < p; j++) {
      let rho = 0;
      for (let k = 0; k < n; k++) rho += Z[k][j] * r[k];
      rho = rho / n + sq[j] * w[j];
      const next = softThreshold(rho, lam) / sq[j];
      const d = next - w[j];
      if (d !== 0) {
        for (let k = 0; k < n; k++) r[k] -= Z[k][j] * d;
        w[j] = next;
        change = Math.max(change, Math.abs(d));
      }
    }
    if (change < tol) break;
  }
  return { w, sweeps: sweeps + 1 };
}

export type Penalty = "ridge" | "lasso";

export const lambdaGrid = (n = 45, lo = -5, hi = 0.5) => Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1)));

/** Coefficients for every λ (lasso warm-started from the largest λ down). */
export function pathFor(xs: number[], ys: number[], pen: Penalty, lams: number[]) {
  const { Z, transform } = polyDesign(xs);
  const ybar = ys.reduce((a, b) => a + b, 0) / ys.length;
  const yc = ys.map((v) => v - ybar);
  const ws: number[][] = new Array(lams.length);
  let warm: number[] | undefined;
  for (let i = lams.length - 1; i >= 0; i--) {
    ws[i] = pen === "ridge" ? ridgeFit(Z, yc, lams[i]) : (warm = lassoFit(Z, yc, lams[i], warm).w);
  }
  return { ws, predict: (w: number[], v: number) => ybar + dot(transform(v), w) };
}

/** K-fold cross-validated MSE for each λ (fold f holds out samples with index % K === f). */
export function cvCurve(xs: number[], ys: number[], pen: Penalty, lams: number[], K = 4): number[] {
  const err = new Array(lams.length).fill(0);
  for (let f = 0; f < K; f++) {
    const tr = xs.map((_, i) => i).filter((i) => i % K !== f);
    const te = xs.map((_, i) => i).filter((i) => i % K === f);
    const fit = pathFor(tr.map((i) => xs[i]), tr.map((i) => ys[i]), pen, lams);
    fit.ws.forEach((w, l) => te.forEach((i) => (err[l] += (fit.predict(w, xs[i]) - ys[i]) ** 2)));
  }
  return err.map((e) => r6(e / xs.length));
}

export function regPath(pen: Penalty, lams = lambdaGrid()) {
  const { xs, ys } = fuelData();
  const fit = pathFor(xs, ys, pen, lams);
  const train = fit.ws.map((w) => r6(xs.reduce((s, v, i) => s + (fit.predict(w, v) - ys[i]) ** 2, 0) / xs.length));
  return { xs, ys, lams, ws: fit.ws.map((w) => w.map(r6)), predict: fit.predict, train, cv: cvCurve(xs, ys, pen, lams) };
}
