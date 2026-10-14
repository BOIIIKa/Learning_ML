import { useMemo, useState } from "react";
import { gradientDescent, leastSquares, studyData } from "../../../engines/regression";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "你来画一条线：什么叫“最好”？", en: "Draw a line yourself: what does “best” mean?" },
  mse: { zh: "平均平方误差", en: "Mean squared error" },
  mae: { zh: "平均绝对误差", en: "Mean absolute error" },
  ls: { zh: "按平方误差最优", en: "Best by squared error" },
  lad: { zh: "按绝对误差最优", en: "Best by absolute error" },
  x: { zh: "每周学习时长（小时）", en: "Study hours per week" },
  y: { zh: "期末成绩", en: "Final score" },
  caption: {
    zh: "拖动两个橙色把手画一条直线，黄色竖线是每个学生的残差。红圈是四位缺考、记了 0 分的学生。点两个按钮对比：按平方误差最优的线被缺考学生明显拉低，按绝对误差最优的线几乎不受影响——“最好”取决于你怎么定义损失。数据为合成示意。",
    en: "Drag the two orange handles to draw a line; the yellow sticks are each student's residual. Red rings are four students who missed the exam and scored 0. Compare the two buttons: the line that is best by squared error is dragged down by the absentees, the one best by absolute error barely moves. “Best” depends on how you define the loss. Synthetic illustrative data.",
  },
};

export default function DrawYourLine({ lang }: { lang: Lang }) {
  const { xs, ys } = useMemo(() => studyData(), []);
  const [p1, setP1] = useState<[number, number]>([1, 70]);
  const [p2, setP2] = useState<[number, number]>([9, 55]);
  const slope = (p2[1] - p1[1]) / (p2[0] - p1[0] || 1e-9);
  const f = (x: number) => p1[1] + slope * (x - p1[0]);
  const mse = xs.reduce((s, x, i) => s + (f(x) - ys[i]) ** 2, 0) / xs.length;
  const mae = xs.reduce((s, x, i) => s + Math.abs(f(x) - ys[i]), 0) / xs.length;

  const snap = (w: number[]) => {
    const g = (x: number) => w[0] + w[1] * ((x - 5) / 5);
    setP1([1, g(1)]);
    setP2([9, g(9)]);
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <CurvePlot xs={xs} ys={ys} outliers={[0, 1, 2, 3]} xDomain={[0, 10]} yDomain={[-5, 105]} xLabel={TEXT.x[lang]} yLabel={TEXT.y[lang]}
        curves={[{ f, color: "var(--class-pos)" }]} residualsTo={f}
        handles={[
          { p: p1, onDrag: ([x, y]) => setP1([Math.min(x, p2[0] - 0.5), y]) },
          { p: p2, onDrag: ([x, y]) => setP2([Math.max(x, p1[0] + 0.5), y]) },
        ]} />
      <div className="w-row">
        <button className="btn" onClick={() => snap(leastSquares("linear", xs, ys))}>{TEXT.ls[lang]}</button>
        <button className="btn" onClick={() => snap(gradientDescent("absolute", "linear", xs, ys, 4000).ws.at(-1)!)}>{TEXT.lad[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.mse[lang]} value={mse.toFixed(1)} tone="accent" />
        <Stat label={TEXT.mae[lang]} value={mae.toFixed(2)} />
      </div>
    </Widget>
  );
}
