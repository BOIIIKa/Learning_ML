import { contours } from "d3-contour";
import { useMemo, useState } from "react";
import { RACERS, SURFACES, run, type OptName, type Vec } from "../../../engines/optimization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider } from "../../ui/Widget";
import "./ch16.css";

type Kind = keyof typeof SURFACES;
const KINDS: Kind[] = ["valley", "rosenbrock", "saddle"];
const STEPS = 150;

const TEXT = {
  title: { zh: "优化器赛跑", en: "Optimizer race" },
  valley: { zh: "病态山谷", en: "Narrow valley" },
  rosenbrock: { zh: "Rosenbrock 香蕉谷", en: "Rosenbrock banana" },
  saddle: { zh: "鞍点", en: "Saddle" },
  mult: { zh: "学习率倍数", en: "Learning-rate multiplier" },
  steps: { zh: "已走步数", en: "Steps taken" },
  names: {
    gd: { zh: "梯度下降", en: "GD" }, momentum: { zh: "动量", en: "Momentum" }, nesterov: { zh: "Nesterov", en: "Nesterov" }, adam: { zh: "Adam", en: "Adam" },
  } as Record<string, Record<Lang, string>>,
  gap: { zh: "f − f*", en: "f − f*" },
  blown: { zh: "发散", en: "diverged" },
  click: { zh: "点击图面可以换起点。", en: "Click the plot to move the start." },
  caption: {
    zh: "四个优化器从同一点出发，走 150 步。病态山谷（κ = 25，斜放 30°）里，梯度下降沿谷底慢慢爬，动量和 Nesterov 借惯性先到；Rosenbrock 的弯曲谷底上，梯度下降 150 步还没到 (1, 1)；鞍点 (0, 0) 附近梯度几乎为零，梯度下降要在那里停留 60 多步，Adam 按坐标缩放梯度，10 步就逃出来了。在山谷或香蕉谷上把学习率倍数调到 ×4：Nesterov 最先发散，梯度下降在陡壁之间来回弹跳。",
    en: "Four optimizers start from the same point and take 150 steps. In the narrow valley (κ = 25, tilted by 30°), gradient descent crawls along the floor while momentum and Nesterov ride their inertia and arrive first. On Rosenbrock's curved floor, gradient descent has not reached (1, 1) after 150 steps. Near the saddle (0, 0) the gradient is almost zero and gradient descent lingers for more than 60 steps; Adam rescales each coordinate and escapes in 10. On the valley or the banana, turn the multiplier up to ×4: Nesterov diverges first and gradient descent bounces between the steep walls.",
  },
};

const COLORS: Record<OptName, string> = { gd: "var(--text-muted)", momentum: "var(--accent-2)", nesterov: "var(--success)", adam: "var(--accent)", rmsprop: "", adamw: "" };
const W = 460, H = 300, NX = 92, NY = 60;
const f1 = (v: number) => v.toFixed(1);

export default function OptimizerRace({ lang }: { lang: Lang }) {
  const [kind, setKind] = useState<Kind>("valley");
  const [m, setM] = useState(0);
  const [shown, setShown] = useState(STEPS);
  const [start, setStart] = useState<Vec | null>(null);
  const S = SURFACES[kind];
  const { x: [x0, x1], y: [y0, y1] } = S.domain;
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * W;
  const sy = (y: number) => H - ((y - y0) / (y1 - y0)) * H;
  const from = start ?? S.start;
  const mult = 2 ** m;
  const runs = useMemo(() => RACERS.map((name) => {
    const path = run(name, { lr: (name === "adam" ? S.lr.adam : S.lr.gd) * mult }, S.grad, from, STEPS);
    return { name, path, diverged: path.length <= STEPS };
  }), [S, mult, from]);
  // Contours of log(f − f* + 0.01), evenly spaced.
  const lines = useMemo(() => {
    const vals: number[] = [];
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const x = x0 + ((x1 - x0) * (i + 0.5)) / NX, y = y1 - ((y1 - y0) * (j + 0.5)) / NY;
      vals.push(Math.log(S.f([x, y]) - S.fmin + 0.01));
    }
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const th = Array.from({ length: 12 }, (_, k) => lo + ((hi - lo) * (k + 0.5)) / 12);
    return contours().size([NX, NY]).thresholds(th)(vals).map((c) =>
      (c.coordinates as number[][][][]).flatMap((poly) => poly.map((ring) => "M" + ring.map(([gx, gy]) => `${f1((gx / NX) * W)},${f1((gy / NY) * H)}`).join("L") + "Z")).join(""));
  }, [S, x0, x1, y0, y1]);
  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(e.currentTarget.getScreenCTM()!.inverse());
    setStart([+(x0 + (p.x / W) * (x1 - x0)).toFixed(3), +(y0 + ((H - p.y) / H) * (y1 - y0)).toFixed(3)]);
  };
  return (
    <Widget title={TEXT.title[lang]} caption={<>{TEXT.caption[lang]} {TEXT.click[lang]}</>} tag="interactive">
      <div className="chips">
        {KINDS.map((k) => <button key={k} className={`chip${kind === k ? " on" : ""}`} onClick={() => { setKind(k); setStart(null); }}>{TEXT[k][lang]}</button>)}
      </div>
      <svg className="lm-svg op-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerDown={pick}>
        <rect x={0} y={0} width={W} height={H} className="op-frame" />
        {lines.map((d, i) => <path key={i} d={d} className="op-contour" />)}
        {runs.map((r) => {
          const pts = r.path.slice(0, shown + 1).filter((p) => p.every((v) => Math.abs(v) < 50));
          return <polyline key={r.name} points={pts.map((p) => `${f1(sx(p[0]))},${f1(sy(p[1]))}`).join(" ")} className="op-race" stroke={COLORS[r.name]} />;
        })}
        {runs.map((r) => {
          const p = r.path[Math.min(shown, r.path.length - 1)];
          return Math.abs(p[0]) < 50 && Math.abs(p[1]) < 50 ? <circle key={r.name} cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={5} fill={COLORS[r.name]} className="op-head" /> : null;
        })}
        <circle cx={f1(sx(from[0]))} cy={f1(sy(from[1]))} r={4} className="op-start" />
      </svg>
      <Slider label={TEXT.mult[lang]} value={m} min={-2} max={2} step={0.25} onChange={setM} format={(v) => `×${(2 ** v).toFixed(2).replace(/\.?0+$/, "")}`} />
      <Slider label={TEXT.steps[lang]} value={shown} min={0} max={STEPS} step={1} onChange={setShown} />
      <div className="op-legend">
        {runs.map((r) => {
          const t = Math.min(shown, r.path.length - 1);
          const gap = S.f(r.path[t]) - S.fmin;
          return (
            <span key={r.name} className="op-leg">
              <i style={{ background: COLORS[r.name] }} />
              {TEXT.names[r.name][lang]}
              <b>{r.diverged && shown >= r.path.length - 1 ? TEXT.blown[lang] : gap < 1e-4 ? "< 1e-4" : gap.toPrecision(2)}</b>
            </span>
          );
        })}
      </div>
    </Widget>
  );
}
