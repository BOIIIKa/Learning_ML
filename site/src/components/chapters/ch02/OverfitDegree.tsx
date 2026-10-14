import { useMemo, useState } from "react";
import { mse, polyFit, sample, truth } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "越复杂越好吗？", en: "Is more complex always better?" },
  degree: { zh: "多项式次数", en: "Polynomial degree" },
  show: { zh: "显示明年的学生", en: "Show next year's students" },
  train: { zh: "训练误差", en: "Training error" },
  test: { zh: "测试误差（明年）", en: "Test error (next year)" },
  chart: { zh: "log₁₀ 均方误差", en: "log₁₀ MSE" },
  caption: {
    zh: "12 位学生（实心点）用来拟合多项式，虚线是真实规律（这里我们“作弊”地知道它）。次数越高，曲线越能穿过每一个训练点，训练误差一路降到 0；可是打开“明年的学生”（空心点），高次曲线在点与点之间剧烈摆动，测试误差急剧上升。数据为合成示意。",
    en: "Twelve students (solid points) are used to fit a polynomial; the dashed curve is the true rule (which we can cheat and know here). As the degree rises the curve threads every training point and training error drops to 0. But turn on next year's students (hollow points): high-degree curves swing wildly between points and test error shoots up. Synthetic illustrative data.",
  },
};

export default function OverfitDegree({ lang }: { lang: Lang }) {
  const train = useMemo(() => sample(12, 1), []);
  const test = useMemo(() => sample(150, 77), []);
  const [d, setD] = useState(3);
  const [show, setShow] = useState(false);
  const curves = useMemo(() => Array.from({ length: 12 }, (_, k) => polyFit(train.xs, train.ys, k, 0)), [train]);
  const trainErr = curves.map((f) => mse(f, train.xs, train.ys));
  const testErr = curves.map((f) => mse(f, test.xs, test.ys));
  const log = (v: number) => Math.log10(Math.max(v, 1e-3));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <CurvePlot xs={train.xs} ys={train.ys} xDomain={[0, 10]} yDomain={[0, 110]}
        extraPoints={show ? test : undefined}
        curves={[{ f: truth, color: "var(--text-muted)", dashed: true, width: 1.8 }, { f: curves[d], color: "var(--class-pos)" }]} />
      <div className="w-row">
        <Slider label={TEXT.degree[lang]} value={d} min={0} max={11} step={1} onChange={setD} />
        <label className="chip"><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> {TEXT.show[lang]}</label>
      </div>
      <MetricChart zero={false} cursor={d} height={130} series={[
        { key: "tr", label: `log₁₀ ${TEXT.train[lang]}`, values: trainErr.map(log), color: "var(--accent)" },
        { key: "te", label: `log₁₀ ${TEXT.test[lang]}`, values: testErr.map(log), color: "var(--accent-2)", dashed: true },
      ]} />
      <div className="w-stats">
        <Stat label={TEXT.train[lang]} value={trainErr[d].toFixed(1)} />
        <Stat label={TEXT.test[lang]} value={testErr[d] > 1e5 ? testErr[d].toExponential(1) : testErr[d].toFixed(1)} tone="accent" />
      </div>
    </Widget>
  );
}
