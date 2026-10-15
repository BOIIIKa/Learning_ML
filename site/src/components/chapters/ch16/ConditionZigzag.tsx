import { useMemo, useState } from "react";
import { QUAD_START, gdRate, heavyBall, optimalEta, quadF, quadRun, stepsTo } from "../../../engines/optimization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch16.css";

const TEXT = {
  title: { zh: "条件数与之字形", en: "Condition number and zig-zags" },
  kappa: { zh: "条件数 κ = L/μ", en: "Condition number κ = L/μ" },
  eta: { zh: "步长 η（以 1/L 为单位）", en: "Step η (in units of 1/L)" },
  gd: { zh: "梯度下降", en: "Gradient descent" },
  hb: { zh: "重球动量（最优 η、β）", en: "Heavy ball (best η, β)" },
  best: { zh: "η = 2/(L+μ)", en: "η = 2/(L+μ)" },
  rate: { zh: "理论收缩率", en: "Rate in theory" },
  pred: { zh: "理论步数", en: "Steps in theory" },
  meas: { zh: "实测步数（误差 ÷1000）", en: "Steps measured (error ÷1000)" },
  never: { zh: "不收敛", en: "never" },
  caption: {
    zh: "二次碗 f = ½(μx² + Ly²)，L = 1，μ = 1/κ，从 (−3, 1) 出发。梯度下降在每个特征方向上把误差乘以 1 − ηλ：陡的方向（λ = L）要求 η < 2/L，否则来回弹跳、越弹越远；缓的方向（λ = μ）每步只缩小 1 − ημ。两头兼顾的最好步长是 η = 2/(L + μ)，收缩率 (κ − 1)/(κ + 1)，所需步数大约与 κ 成正比。拉大 κ，看轨迹在山谷两壁之间之字形前进。重球动量把收缩率变成 (√κ − 1)/(√κ + 1)：κ = 100 时，步数从 346 降到 52。",
    en: "A quadratic bowl f = ½(μx² + Ly²) with L = 1 and μ = 1/κ, started at (−3, 1). Gradient descent multiplies the error along each eigen-direction by 1 − ηλ. The steep direction (λ = L) needs η < 2/L, or the iterate bounces ever further; the flat one (λ = μ) shrinks by only 1 − ημ per step. The best compromise is η = 2/(L + μ), with rate (κ − 1)/(κ + 1), and the number of steps grows roughly in proportion to κ. Raise κ and watch the path zig-zag between the valley walls. The heavy ball turns the rate into (√κ − 1)/(√κ + 1): at κ = 100 the steps drop from 346 to 52.",
  },
};

const W = 460, H = 230;
const X = 3.4, Y = 1.7;
const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => W / 2 + (x / X) * (W / 2 - 8);
const sy = (y: number) => H / 2 - (y / Y) * (H / 2 - 8);

export default function ConditionZigzag({ lang }: { lang: Lang }) {
  const [kappa, setKappa] = useState(10);
  const [eta, setEta] = useState(1);
  const [hb, setHb] = useState(false);
  const path = useMemo(() => quadRun(kappa, eta, hb ? "momentum" : "gd", 1500), [kappa, eta, hb]);
  const rate = hb ? heavyBall(kappa).rate : gdRate(kappa, eta);
  const pred = rate < 1 ? Math.ceil(Math.log(1e-3) / Math.log(rate)) : Infinity;
  const meas = stepsTo(path);
  const f0 = quadF(kappa)(QUAD_START);
  const levels = Array.from({ length: 9 }, (_, k) => f0 * 0.45 ** k);
  const shown = path.slice(0, 80).filter((p) => Math.abs(p[0]) < 2 * X && Math.abs(p[1]) < 2 * Y);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lm-svg op-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect x={0} y={0} width={W} height={H} className="op-frame" />
        {levels.map((c, i) => (
          <ellipse key={i} cx={f1(sx(0))} cy={f1(sy(0))} rx={f1(Math.sqrt(2 * c * kappa) * (W / 2 - 8) / X)} ry={f1(Math.sqrt(2 * c) * (H / 2 - 8) / Y)} className="op-level" />
        ))}
        <polyline points={shown.map((p) => `${f1(sx(p[0]))},${f1(sy(p[1]))}`).join(" ")} className={hb ? "op-path hb" : "op-path"} />
        {shown.slice(0, 40).map((p, i) => <circle key={i} cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={2.4} className={hb ? "op-dot hb" : "op-dot"} />)}
        <circle cx={f1(sx(0))} cy={f1(sy(0))} r={4} className="op-min" />
      </svg>
      <Slider label={TEXT.kappa[lang]} value={kappa} min={1} max={100} step={1} onChange={setKappa} />
      <Slider label={TEXT.eta[lang]} value={eta} min={0.1} max={2.1} step={0.01} onChange={setEta} format={(v) => v.toFixed(2)} />
      <div className="chips">
        <button className={`chip${!hb ? " on" : ""}`} onClick={() => setHb(false)}>{TEXT.gd[lang]}</button>
        <button className={`chip${hb ? " on" : ""}`} onClick={() => setHb(true)}>{TEXT.hb[lang]}</button>
        <button className="chip" onClick={() => { setHb(false); setEta(+optimalEta(kappa).toFixed(2)); }}>{TEXT.best[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.rate[lang]} value={rate.toFixed(3)} tone={rate >= 1 ? "bad" : undefined} />
        <Stat label={TEXT.pred[lang]} value={Number.isFinite(pred) ? pred : TEXT.never[lang]} />
        <Stat label={TEXT.meas[lang]} value={Number.isFinite(meas) ? meas : TEXT.never[lang]} tone="accent" />
      </div>
    </Widget>
  );
}
