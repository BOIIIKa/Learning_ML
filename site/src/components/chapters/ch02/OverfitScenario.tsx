import { useMemo, useState } from "react";
import { cohort, linspace, mse, polyFit, polyPredict, testSet, truth } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import ErrChart from "./ErrChart";
import FitPlot from "./FitPlot";

const TEXT = {
  title: { zh: "用去年的 12 个学生，预测明年的学生", en: "Fit last year's 12 students, predict next year's" },
  degree: { zh: "多项式阶数 d", en: "Polynomial degree d" },
  truthBtn: { zh: "显示真实规律", en: "Show the true trend" },
  train: { zh: "训练误差（去年 12 人）", en: "Training error (last year's 12)" },
  test: { zh: "测试误差（明年 150 人）", en: "Test error (next year's 150)" },
  gap: { zh: "两者之差", en: "Gap" },
  x: { zh: "每周学习时长（小时）", en: "Study hours per week" },
  y: { zh: "期末成绩", en: "Final score" },
  dx: { zh: "阶数 d", en: "degree d" },
  ey: { zh: "均方误差", en: "MSE" },
  caption: {
    zh: "实心点是去年的 12 位学生（用来拟合），空心点是明年的 150 位学生（拟合时看不到）。拖动阶数：训练误差一路下降，d = 11 时曲线穿过每一个点、训练误差为 0；但测试误差先降后升——阶数太低抓不住“先慢后快再饱和”的形状，太高则开始追逐噪声。数据为合成示意，真实规律可以点按钮查看。",
    en: "Filled dots are last year's 12 students (used for fitting); hollow dots are next year's 150 students (unseen during fitting). Drag the degree: training error keeps falling and reaches 0 at d = 11, where the curve passes through every point. Test error, however, first falls and then rises: too low a degree misses the slow-start, fast-rise, saturating shape; too high a degree starts chasing noise. Synthetic data; the button reveals the true trend.",
  },
};

const GRID = linspace(0, 10, 121);
const DEGREES = Array.from({ length: 12 }, (_, d) => d);

export default function OverfitScenario({ lang }: { lang: Lang }) {
  const train = useMemo(() => cohort(12, 8, 3), []);
  const test = useMemo(() => testSet(150, 8, 99), []);
  const [d, setD] = useState(1);
  const [showTruth, setShowTruth] = useState(false);
  const fits = useMemo(() => DEGREES.map((k) => polyFit(train.xs, train.ys, k)), [train]);
  const errs = useMemo(() => fits.map((w) => ({
    train: mse(train.xs.map((x) => polyPredict(w, x)), train.ys),
    test: mse(test.xs.map((x) => polyPredict(w, x)), test.ys),
  })), [fits, train, test]);
  const w = fits[d];
  const e = errs[d];
  const curves = [{ ys: GRID.map((x) => polyPredict(w, x)), className: "g2-fit", key: "fit" }];
  if (showTruth) curves.unshift({ ys: GRID.map(truth), className: "g2-truth", key: "truth" });

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <FitPlot grid={GRID} xDomain={[0, 10]} yDomain={[10, 100]} xLabel={TEXT.x[lang]} yLabel={TEXT.y[lang]} curves={curves}
        points={[{ ...test, className: "g2-test", r: 3 }, { ...train, className: "g2-train", r: 5 }]} />
      <div className="w-row">
        <Slider label={TEXT.degree[lang]} value={d} min={0} max={11} step={1} onChange={setD} />
        <button className={`chip${showTruth ? " on" : ""}`} onClick={() => setShowTruth(!showTruth)}>{TEXT.truthBtn[lang]}</button>
      </div>
      <ErrChart xs={DEGREES} cursor={d} xLabel={TEXT.dx[lang]} yLabel={TEXT.ey[lang]} height={170} yDomain={[0, 450]}
        format={(v) => (v < 1e4 ? v.toFixed(1) : v.toExponential(1))}
        series={[
          { key: "train", label: TEXT.train[lang], values: errs.map((x) => x.train), color: "var(--class-neg)" },
          { key: "test", label: TEXT.test[lang], values: errs.map((x) => x.test), color: "var(--class-pos)" },
        ]} />
      <div className="w-stats">
        <Stat label={TEXT.train[lang]} value={e.train.toFixed(1)} />
        <Stat label={TEXT.test[lang]} value={e.test < 1e4 ? e.test.toFixed(1) : e.test.toExponential(1)} tone="accent" />
        <Stat label={TEXT.gap[lang]} value={(e.test - e.train) < 1e4 ? (e.test - e.train).toFixed(1) : (e.test - e.train).toExponential(1)} tone={e.test - e.train > 50 ? "bad" : undefined} />
      </div>
    </Widget>
  );
}
