import { contours } from "d3-contour";
import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "梯度下降与条件数", en: "Gradient descent and the condition number" },
  kappa: { zh: "条件数 κ = λmax/λmin", en: "Condition number κ = λmax/λmin" },
  eta: { zh: "步长 η（× 2/λmax）", en: "Step size η (× 2/λmax)" },
  steps: { zh: "到达 f < 10⁻³ 的步数", en: "Steps to reach f < 10⁻³" },
  diverge: { zh: "发散", en: "diverges" },
  caption: {
    zh: "f(w) = ½ wᵀHw，H 的特征值为 1 与 κ。梯度 ∇f = Hw，梯度下降 w ← w − η∇f。η 必须小于 2/λmax 才收敛（超过就发散）；κ 越大，等高线越扁，路径越“之”字形，收敛越慢——收敛率由 (κ−1)/(κ+1) 控制。",
    en: "f(w) = ½ wᵀHw with eigenvalues 1 and κ. The gradient is ∇f = Hw and gradient descent steps w ← w − η∇f. It converges only if η < 2/λmax (beyond that it diverges). The larger κ, the flatter the contours, the more zig-zag the path and the slower the convergence, governed by (κ−1)/(κ+1).",
  },
};

const W = 460;
const H = 340;

export default function GradientBowl({ lang }: { lang: Lang }) {
  const [kappa, setKappa] = useState(8);
  const [etaFrac, setEtaFrac] = useState(0.9);
  const lmax = kappa;
  const eta = (etaFrac * 2) / lmax;
  const sx = scaleLinear().domain([-3.2, 3.2]).range([0, W]);
  const sy = scaleLinear().domain([-2.4, 2.4]).range([H, 0]);
  const fval = (w: number[]) => 0.5 * (w[0] * w[0] + kappa * w[1] * w[1]);

  const { path, steps } = useMemo(() => {
    let w = [-2.9, 1.6];
    const pts = [w];
    let steps = -1;
    for (let k = 1; k <= 400; k++) {
      w = [w[0] - eta * w[0], w[1] - eta * kappa * w[1]];
      if (k <= 60) pts.push(w);
      if (steps < 0 && fval(w) < 1e-3) steps = k;
      if (!Number.isFinite(w[0]) || Math.abs(w[1]) > 1e6) break;
    }
    return { path: pts, steps };
  }, [kappa, eta]);

  const contourPaths = useMemo(() => {
    const n = 80;
    const vals: number[] = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const x = -3.2 + (6.4 * c) / (n - 1);
      const y = -2.4 + (4.8 * r) / (n - 1);
      vals.push(0.5 * (x * x + kappa * y * y));
    }
    const levels = [0.05, 0.2, 0.5, 1, 2, 4, 7, 11, 16];
    return contours().size([n, n]).thresholds(levels)(vals).map((g) =>
      g.coordinates.flatMap((poly) => poly.map((ring) => "M" + ring.map(([gx, gy]) => `${sx(-3.2 + (6.4 * (gx - 0.5)) / (n - 1)).toFixed(1)},${sy(-2.4 + (4.8 * (gy - 0.5)) / (n - 1)).toFixed(1)}`).join("L") + "Z")).join(""),
    );
  }, [kappa]);

  const clip = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const d = "M" + path.map((w) => `${clip(sx(w[0]), -50, W + 50).toFixed(1)},${clip(sy(w[1]), -50, H + 50).toFixed(1)}`).join("L");

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="gb2-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <defs><clipPath id="gb2-clip"><rect width={W} height={H} rx={12} /></clipPath></defs>
        <g clipPath="url(#gb2-clip)">
          <rect width={W} height={H} className="plot-bg" />
          {contourPaths.map((p, i) => <path key={i} d={p} className="gb2-contour" style={{ opacity: 0.25 + 0.07 * (9 - i) }} />)}
          <path d={d} className="gb2-path" />
          {path.map((w, i) => <circle key={i} cx={clip(sx(w[0]), -50, W + 50)} cy={clip(sy(w[1]), -50, H + 50)} r={i === 0 ? 6 : 3} className={i === 0 ? "gb2-start" : "gb2-pt"} />)}
          <circle cx={sx(0)} cy={sy(0)} r={5} className="gb2-min" />
        </g>
      </svg>
      <div className="w-row">
        <Slider label={TEXT.kappa[lang]} value={kappa} min={1} max={30} step={0.5} onChange={setKappa} />
        <Slider label={TEXT.eta[lang]} value={etaFrac} min={0.05} max={1.1} step={0.01} onChange={setEtaFrac} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.steps[lang]} value={steps > 0 ? steps : etaFrac >= 1 ? TEXT.diverge[lang] : "> 400"} tone={steps > 0 ? "accent" : "bad"} />
        <Stat label="(κ−1)/(κ+1)" value={((kappa - 1) / (kappa + 1)).toFixed(3)} />
      </div>
    </Widget>
  );
}
