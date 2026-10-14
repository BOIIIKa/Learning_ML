import { useEffect, useMemo, useState } from "react";
import { gradientDescent, leastSquares, predict, randomSearch, risk, studyData, type Loss, type Model } from "../../../engines/regression";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Stat } from "../../ui/Widget";

type Algo = "closed" | "gd" | "random";

const TEXT = {
  title: { zh: "三要素组合器：模型 × 策略 × 算法", en: "Triad composer: model × strategy × algorithm" },
  model: { zh: "模型", en: "Model" },
  loss: { zh: "策略（损失）", en: "Strategy (loss)" },
  algo: { zh: "算法", en: "Algorithm" },
  constant: { zh: "常数", en: "Constant" },
  linear: { zh: "直线", en: "Line" },
  cubic: { zh: "三次多项式", en: "Cubic" },
  squared: { zh: "平方", en: "Squared" },
  absolute: { zh: "绝对值", en: "Absolute" },
  huber: { zh: "Huber", en: "Huber" },
  closed: { zh: "闭式解", en: "Closed form" },
  gd: { zh: "梯度下降", en: "Gradient descent" },
  random: { zh: "随机搜索", en: "Random search" },
  noClosed: { zh: "闭式解只适用于平方损失", en: "A closed form exists only for squared loss" },
  risk: { zh: "经验风险", en: "Empirical risk" },
  iter: { zh: "迭代", en: "Iteration" },
  replay: { zh: "↻ 重放", en: "↻ Replay" },
  caption: {
    zh: "三个下拉选项相互独立：任意组合都是一个学习算法。试试：常数 + 平方 → 均值；常数 + 绝对值 → 中位数；直线 + 平方 + 闭式解 → 最小二乘；绝对值 + 梯度下降 → 对缺考学生更稳健的拟合。下方曲线是训练过程中的经验风险。",
    en: "The three choices are independent; every combination is a learning algorithm. Try: constant + squared → the mean; constant + absolute → the median; line + squared + closed form → least squares; absolute + gradient descent → a fit that is robust to the absentees. The chart shows empirical risk during training.",
  },
};

export default function TriadComposer({ lang }: { lang: Lang }) {
  const { xs, ys } = useMemo(() => studyData(), []);
  const [model, setModel] = useState<Model>("linear");
  const [loss, setLoss] = useState<Loss>("squared");
  const [algo, setAlgo] = useState<Algo>("gd");
  const [t, setT] = useState(0);
  const [run, setRun] = useState(0);

  const closedOk = loss === "squared";
  const effAlgo: Algo = algo === "closed" && !closedOk ? "gd" : algo;
  const path = useMemo(() => {
    if (effAlgo === "closed") {
      const w = leastSquares(model, xs, ys);
      return { ws: [w], risks: [risk(loss, model, w, xs, ys)] };
    }
    return effAlgo === "gd" ? gradientDescent(loss, model, xs, ys, 150) : randomSearch(loss, model, xs, ys, 150, run + 1);
  }, [model, loss, effAlgo, xs, ys, run]);

  useEffect(() => {
    setT(0);
    if (path.ws.length === 1) return;
    let k = 0;
    const id = setInterval(() => {
      k = Math.min(path.ws.length - 1, k + 2);
      setT(k);
      if (k >= path.ws.length - 1) clearInterval(id);
    }, 40);
    return () => clearInterval(id);
  }, [path]);

  const w = path.ws[Math.min(t, path.ws.length - 1)];
  const f = (x: number) => predict(model, w, x);
  const chip = <K extends string>(value: K, cur: K, set: (v: K) => void, label: string, disabled = false, title?: string) => (
    <button key={value} className={`chip${value === cur ? " on" : ""}`} disabled={disabled} title={title} onClick={() => set(value)}>{label}</button>
  );

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="triad-pickers">
        <div><span className="tp-k">{TEXT.model[lang]}</span><div className="chips">{(["constant", "linear", "cubic"] as Model[]).map((m) => chip(m, model, setModel, TEXT[m][lang]))}</div></div>
        <div><span className="tp-k">{TEXT.loss[lang]}</span><div className="chips">{(["squared", "absolute", "huber"] as Loss[]).map((l) => chip(l, loss, setLoss, TEXT[l][lang]))}</div></div>
        <div><span className="tp-k">{TEXT.algo[lang]}</span><div className="chips">
          {chip("closed" as Algo, effAlgo, setAlgo, TEXT.closed[lang], !closedOk, closedOk ? undefined : TEXT.noClosed[lang])}
          {chip("gd" as Algo, effAlgo, setAlgo, TEXT.gd[lang])}
          {chip("random" as Algo, effAlgo, setAlgo, TEXT.random[lang])}
        </div></div>
      </div>
      <CurvePlot xs={xs} ys={ys} outliers={[0, 1, 2, 3]} xDomain={[0, 10]} yDomain={[-5, 105]} curves={[{ f, color: "var(--class-pos)" }]} residualsTo={f} height={290} />
      {path.risks.length > 1 && <MetricChart series={[{ key: "r", label: TEXT.risk[lang], values: path.risks, color: "var(--accent)" }]} cursor={Math.min(t, path.risks.length - 1)} height={120} />}
      <div className="w-row">
        <button className="btn" onClick={() => setRun(run + 1)}>{TEXT.replay[lang]}</button>
        {!closedOk && algo === "closed" && <span className="muted">{TEXT.noClosed[lang]}</span>}
      </div>
      <div className="w-stats">
        <Stat label={TEXT.risk[lang]} value={path.risks[Math.min(t, path.risks.length - 1)].toFixed(2)} tone="accent" />
        <Stat label={TEXT.iter[lang]} value={path.ws.length === 1 ? "—" : `${t} / ${path.ws.length - 1}`} />
      </div>
    </Widget>
  );
}
