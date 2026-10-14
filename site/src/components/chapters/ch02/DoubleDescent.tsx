import { useEffect, useMemo, useState } from "react";
import { cohort, DD_PS, doubleDescent, linspace, randomFeatures, rfFit, rfPredict, truth, type DdPoint } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import ErrChart from "./ErrChart";
import FitPlot from "./FitPlot";

const TEXT = {
  title: { zh: "双下降：越过插值阈值 p = n", en: "Double descent: crossing the interpolation threshold p = n" },
  p: { zh: "随机特征数 p", en: "Random features p" },
  ridge: { zh: "加一点岭正则（λ = 0.01）", en: "Add a little ridge (λ = 0.01)" },
  train: { zh: "训练误差", en: "Training error" },
  test: { zh: "测试误差（中位数）", en: "Test error (median)" },
  norm: { zh: "‖w‖（这一次）", en: "‖w‖ (this draw)" },
  busy: { zh: "正在扫描 p = 1 … 1000…", en: "Sweeping p = 1 … 1000…" },
  x: { zh: "每周学习时长（小时）", en: "Study hours per week" },
  y: { zh: "成绩", en: "Score" },
  px: { zh: "特征数 p（对数）", en: "features p (log)" },
  ey: { zh: "均方误差（对数）", en: "MSE (log)" },
  caption: {
    zh: "20 位学生（噪声较小，σ = 4），模型是 p 个随机余弦特征 cos(ωu + b) 的线性组合，用最小二乘拟合；p > n 时有无穷多个解能穿过所有点，取其中范数最小的那个。下图是 9 次随机抽取特征的中位数。p 接近 n = 20 时，模型“刚好”能穿过所有点，只能用很大的系数硬凑，曲线剧烈震荡，测试误差冲到顶峰；p 继续增大，最小范数解反而越来越平滑，测试误差第二次下降。打开岭正则，峰就消失了。",
    en: "Twenty students (less noise, σ = 4). The model is a linear combination of p random cosine features cos(ωu + b), fitted by least squares; when p > n infinitely many solutions pass through every point and we take the one with the smallest norm. The chart shows the median over 9 random feature draws. Near p = n = 20 the model can only just pass through every point, which forces huge coefficients: the curve oscillates wildly and test error peaks. As p keeps growing, the minimum-norm solution gets smoother and test error falls a second time. Switch on a little ridge and the peak disappears.",
  },
};

const GRID = linspace(0.5, 9.5, 161);
const N = 20;

export default function DoubleDescent({ lang }: { lang: Lang }) {
  const data = useMemo(() => cohort(N, 4, 1), []);
  const [idx, setIdx] = useState(DD_PS.indexOf(22));
  const [ridge, setRidge] = useState(false);
  const [curves, setCurves] = useState<Record<string, DdPoint[]>>({});
  const [mounted, setMounted] = useState(false);
  const lam = ridge ? 0.01 : 0;
  const key = String(lam);

  useEffect(() => setMounted(true), []);
  // Near p = n the solves are so ill-conditioned that the last bit of Math.cos matters, so everything here
  // runs in the browser only (no server-rendered numbers to mismatch).
  useEffect(() => {
    if (curves[key]) return;
    const id = setTimeout(() => setCurves((c) => ({ ...c, [key]: doubleDescent(data.xs, data.ys, 4, lam) })), 30);
    return () => clearTimeout(id);
  }, [key, lam, curves, data]);

  const p = DD_PS[idx];
  const model = useMemo(() => (mounted ? rfFit(randomFeatures(p, 7919 * p + 17), data.xs, data.ys, lam) : null), [mounted, p, lam, data]);
  const curve = curves[key];
  const pt = curve?.[idx];
  const norm = model ? Math.sqrt(model.w.reduce((a, b) => a + b * b, 0)) : NaN;
  const fmt = (v: number) => (v <= 0.1 ? "≈ 0" : v < 1e4 ? v.toFixed(1) : v.toExponential(1));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <FitPlot grid={GRID} xDomain={[0, 10]} yDomain={[10, 100]} xLabel={TEXT.x[lang]} yLabel={TEXT.y[lang]} height={240}
        curves={[{ ys: GRID.map(truth), className: "g2-truth", key: "t" }, ...(model ? [{ ys: GRID.map((x) => rfPredict(model, x)), className: "g2-fit", key: "f" }] : [])]}
        points={[{ ...data, className: "g2-train" }]} />
      <div className="w-row">
        <Slider label={TEXT.p[lang]} value={idx} min={0} max={DD_PS.length - 1} step={1} onChange={setIdx} format={() => String(p)} />
        <button className={`chip${ridge ? " on" : ""}`} onClick={() => setRidge(!ridge)}>{TEXT.ridge[lang]}</button>
      </div>
      {curve ? (
        <ErrChart xs={DD_PS} xLog yLog yDomain={[0.1, 1e5]} cursor={idx} xLabel={TEXT.px[lang]} yLabel={TEXT.ey[lang]} height={190} format={fmt}
          vlines={[{ x: N, label: "p = n", color: "var(--accent-3)" }]}
          series={[
            { key: "train", label: TEXT.train[lang], values: curve.map((c) => Math.max(c.train, 0.1)), color: "var(--class-neg)" },
            { key: "test", label: TEXT.test[lang], values: curve.map((c) => c.test), color: "var(--class-pos)", width: 3 },
          ]} />
      ) : (
        <div className="g2-busy">{TEXT.busy[lang]}</div>
      )}
      <div className="w-stats">
        <Stat label="p / n" value={`${p} / ${N}`} />
        <Stat label={TEXT.train[lang]} value={pt ? fmt(pt.train) : "…"} />
        <Stat label={TEXT.test[lang]} value={pt ? fmt(pt.test) : "…"} tone="accent" />
        <Stat label={TEXT.norm[lang]} value={Number.isFinite(norm) ? (norm < 1e4 ? norm.toFixed(1) : norm.toExponential(1)) : "…"} />
      </div>
    </Widget>
  );
}
