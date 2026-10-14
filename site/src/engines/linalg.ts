// Tiny 2-D linear algebra for the chapter 00 visuals.
export type Vec = [number, number];
export type Mat = [[number, number], [number, number]];

export const mul = (A: Mat, v: Vec): Vec => [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];
export const matmul = (A: Mat, B: Mat): Mat => [
  [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
  [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
];
export const transpose = (A: Mat): Mat => [[A[0][0], A[1][0]], [A[0][1], A[1][1]]];
export const det = (A: Mat) => A[0][0] * A[1][1] - A[0][1] * A[1][0];
export const rot = (theta: number): Mat => [[Math.cos(theta), -Math.sin(theta)], [Math.sin(theta), Math.cos(theta)]];
export const diag = (a: number, b: number): Mat => [[a, 0], [0, b]];
export const lerpMat = (A: Mat, B: Mat, t: number): Mat => [
  [A[0][0] + (B[0][0] - A[0][0]) * t, A[0][1] + (B[0][1] - A[0][1]) * t],
  [A[1][0] + (B[1][0] - A[1][0]) * t, A[1][1] + (B[1][1] - A[1][1]) * t],
];

/** Eigen-decomposition of a symmetric 2×2 matrix: values descending, unit vectors. */
export function eigSym(S: Mat): { values: Vec; vectors: [Vec, Vec] } {
  const [a, b, c] = [S[0][0], S[0][1], S[1][1]];
  const m = (a + c) / 2;
  const r = Math.hypot((a - c) / 2, b);
  const theta = 0.5 * Math.atan2(2 * b, a - c);
  const v1: Vec = [Math.cos(theta), Math.sin(theta)];
  const v2: Vec = [-Math.sin(theta), Math.cos(theta)];
  return { values: [m + r, m - r], vectors: [v1, v2] };
}

/**
 * SVD A = U Σ Vᵀ written as rotations: A = R(φ) · diag(σ1, ±σ2) · R(−θ).
 * A reflection (det < 0) is absorbed into the sign of the second singular value.
 */
export function svd2(A: Mat): { theta: number; sigma: Vec; phi: number } {
  const AtA = matmul(transpose(A), A);
  const { values, vectors } = eigSym(AtA);
  const s1 = Math.sqrt(Math.max(0, values[0]));
  const s2 = Math.sqrt(Math.max(0, values[1])) * (det(A) < 0 ? -1 : 1);
  const theta = Math.atan2(vectors[0][1], vectors[0][0]);
  // U's first column is A v1 / σ1 (or any unit vector when A = 0).
  const u = mul(A, vectors[0]);
  const phi = s1 > 1e-12 ? Math.atan2(u[1], u[0]) : 0;
  return { theta, sigma: [s1, s2], phi };
}

export const fromSvd = ({ theta, sigma, phi }: ReturnType<typeof svd2>): Mat =>
  matmul(matmul(rot(phi), diag(sigma[0], sigma[1])), rot(-theta));

/** Maximum-likelihood mean and covariance (divides by n). */
export function gaussianMle(X: number[][]): { mean: Vec; cov: Mat } {
  const n = X.length;
  const mean: Vec = [X.reduce((s, p) => s + p[0], 0) / n, X.reduce((s, p) => s + p[1], 0) / n];
  let sxx = 0, sxy = 0, syy = 0;
  for (const [x, y] of X) {
    sxx += (x - mean[0]) ** 2;
    sxy += (x - mean[0]) * (y - mean[1]);
    syy += (y - mean[1]) ** 2;
  }
  return { mean, cov: [[sxx / n, sxy / n], [sxy / n, syy / n]] };
}

/** Average log-likelihood of X under N(mean, cov). */
export function gaussianLogLik(X: number[][], mean: Vec, cov: Mat): number {
  const d = det(cov);
  if (d <= 0) return -Infinity;
  const inv: Mat = [[cov[1][1] / d, -cov[0][1] / d], [-cov[1][0] / d, cov[0][0] / d]];
  let s = 0;
  for (const p of X) {
    const v: Vec = [p[0] - mean[0], p[1] - mean[1]];
    const q = mul(inv, v);
    s += -0.5 * (v[0] * q[0] + v[1] * q[1]);
  }
  return s / X.length - 0.5 * Math.log(d) - Math.log(2 * Math.PI);
}

/** argmin ‖x − t‖² subject to a·x ≤ c, with its KKT multiplier. */
export function projectHalfplane(t: Vec, a: Vec, c: number): { x: Vec; lambda: number; active: boolean } {
  const viol = a[0] * t[0] + a[1] * t[1] - c;
  if (viol <= 0) return { x: t, lambda: 0, active: false };
  const aa = a[0] ** 2 + a[1] ** 2;
  // Stationarity: 2(x − t) + λa = 0 with a·x = c  ⇒  λ = 2·viol/‖a‖².
  return { x: [t[0] - (viol / aa) * a[0], t[1] - (viol / aa) * a[1]], lambda: (2 * viol) / aa, active: true };
}
