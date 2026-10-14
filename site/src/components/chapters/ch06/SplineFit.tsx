import { useMemo, useState } from "react";
import { deliveryData, deliveryTrend, evenKnots, fitBasis, knnRegress, rmse, type BasisKind } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch06.css";

type Mode = "knn" | BasisKind;

const TEXT = {
  title: { zh: "基函数扩展：从局部平均到样条", en: "Basis expansions: from local averages to splines" },
  knn: { zh: "kNN 平均", en: "kNN average" },
  poly: { zh: "全局多项式", en: "Global polynomial" },
  step: { zh: "分段常数", en: "Piecewise constant" },
  linear: { zh: "线性样条", en: "Linear spline" },
  cubic: { zh: "三次样条", en: "Cubic spline" },
  k: { zh: "邻居数 k", en: "Neighbours k" },
  degree: { zh: "多项式次数", en: "Polynomial degree" },
  knots: { zh: "节点数 K", en: "Knots K" },
  lambda: { zh: "平滑惩罚 λ", en: "Smoothing penalty λ" },
  df: { zh: "自由度（有效参数个数）", en: "Degrees of freedom" },
  train: { zh: "训练 RMSE（分钟）", en: "Training RMSE (min)" },
  test: { zh: "测试 RMSE（分钟）", en: "Test RMSE (min)" },
  xl: { zh: "配送距离（km）", en: "Distance (km)" },
  yl: { zh: "送达用时（分钟）", en: "Delivery time (min)" },
  caption: {
    zh: "60 个订单的配送距离与用时，灰色虚线是真实趋势。kNN 把每个位置的预测取为最近 k 单的平均，结果是阶梯状的；基函数扩展则把 x 变成 h₁(x), …, hₚ(x)，再做线性回归。全局多项式次数一高，两端就剧烈摆动；样条在节点（竖虚线）处把多项式“缝”起来，局部灵活又不失控。三次样条有 K + 4 个自由度；加上平滑惩罚 λ，有效自由度连续地变小。",
    en: "Distance and time for 60 deliveries; the grey dashed curve is the true trend. kNN predicts the average of the k nearest orders, which gives a staircase. A basis expansion maps x to h₁(x), …, hₚ(x) and runs linear regression on them. A high-degree global polynomial swings wildly at the ends; a spline stitches low-degree pieces together at the knots (dashed verticals), flexible locally but under control. A cubic spline has K + 4 degrees of freedom; a smoothing penalty λ shrinks the effective degrees of freedom continuously.",
  },
};

const MODES: Mode[] = ["knn", "poly", "step", "linear", "cubic"];

export default function SplineFit({ lang }: { lang: Lang }) {
  const [mode, setMode] = useState<Mode>("cubic");
  const [k, setK] = useState(7);
  const [degree, setDegree] = useState(3);
  const [K, setKnots] = useState(5);
  const [logLam, setLogLam] = useState(-6);
  const train = useMemo(() => deliveryData(60, 3), []);
  const test = useMemo(() => deliveryData(300, 77), []);
  const knots = mode === "knn" || mode === "poly" ? [] : evenKnots(K);
  const lambda = logLam <= -6 ? 0 : 10 ** logLam;

  const model = useMemo(() => {
    if (mode === "knn") {
      const f = (x: number) => knnRegress(train.xs, train.ys, x, k);
      return { f, df: train.xs.length / k };
    }
    const fit = fitBasis(train.xs, train.ys, mode, degree, knots, lambda);
    return { f: fit.predict, df: fit.df };
  }, [mode, k, degree, K, lambda, train]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {MODES.map((m) => (
          <button key={m} className={`chip${m === mode ? " on" : ""}`} onClick={() => setMode(m)}>{TEXT[m][lang]}</button>
        ))}
      </div>
      <CurvePlot xs={train.xs} ys={train.ys} xDomain={[0, 10]} yDomain={[5, 52]} xLabel={TEXT.xl[lang]} yLabel={TEXT.yl[lang]}
        curves={[
          { f: deliveryTrend, color: "var(--text-muted)", width: 1.6, dashed: true },
          { f: model.f, color: "var(--accent-3)", width: 3 },
        ]}
        vlines={knots.map((x) => ({ x, color: "var(--border)" }))} />
      <div className="w-row">
        {mode === "knn" && <Slider label={TEXT.k[lang]} value={k} min={1} max={60} step={1} onChange={setK} />}
        {mode === "poly" && <Slider label={TEXT.degree[lang]} value={degree} min={1} max={12} step={1} onChange={setDegree} />}
        {mode !== "knn" && mode !== "poly" && (
          <>
            <Slider label={TEXT.knots[lang]} value={K} min={1} max={15} step={1} onChange={setKnots} />
            <Slider label={TEXT.lambda[lang]} value={logLam} min={-6} max={1} step={0.1} onChange={setLogLam}
              format={(v) => (v <= -6 ? "0" : `10^${v.toFixed(1)}`)} />
          </>
        )}
      </div>
      <div className="w-stats">
        <Stat label={TEXT.df[lang]} value={model.df.toFixed(1)} />
        <Stat label={TEXT.train[lang]} value={rmse(model.f, train.xs, train.ys).toFixed(2)} />
        <Stat label={TEXT.test[lang]} value={rmse(model.f, test.xs, test.ys).toFixed(2)} tone="accent" />
      </div>
    </Widget>
  );
}
