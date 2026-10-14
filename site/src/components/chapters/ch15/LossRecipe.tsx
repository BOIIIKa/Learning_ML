import { useMemo, useState } from "react";
import { DISTS, type DistId } from "../../../engines/losses-as-likelihood";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch15.css";

type L2 = { zh: string; en: string };
const D: { id: DistId; name: L2; q: L2; head: string; loss: string; classic: L2; y: { min: number; max: number; step: number; init: number }; f: [number, number] }[] = [
  { id: "gaussian", name: { zh: "高斯", en: "Gaussian" }, q: { zh: "成绩是多少？", en: "What score?" }, head: "μ = f", loss: "ℓ = ½ (y − f)² + ½ log 2π",
    classic: { zh: "平方损失 → 第 03 章的最小二乘线性回归", en: "Squared loss → least-squares regression (chapter 03)" }, y: { min: -2, max: 2, step: 0.1, init: 1 }, f: [-4, 4] },
  { id: "bernoulli", name: { zh: "伯努利", en: "Bernoulli" }, q: { zh: "能及格吗？", en: "Pass or fail?" }, head: "p = σ(f)", loss: "ℓ = log(1 + eᶠ) − y·f",
    classic: { zh: "二元交叉熵 → 第 04 章的逻辑回归", en: "Binary cross-entropy → logistic regression (chapter 04)" }, y: { min: 0, max: 1, step: 1, init: 1 }, f: [-5, 5] },
  { id: "categorical", name: { zh: "分类分布", en: "Categorical" }, q: { zh: "得 A、B 还是 C？", en: "Grade A, B or C?" }, head: "p = softmax(f, 0, 0)", loss: "ℓ = log Σₖ e^{fₖ} − f_y",
    classic: { zh: "softmax 交叉熵 → 第 04 章的多类逻辑回归 / 最大熵", en: "Softmax cross-entropy → multinomial logistic / max-ent (chapter 04)" }, y: { min: 0, max: 1, step: 1, init: 0 }, f: [-5, 5] },
  { id: "laplace", name: { zh: "拉普拉斯", en: "Laplace" }, q: { zh: "成绩（有离群点）？", en: "Score, with outliers?" }, head: "m = f", loss: "ℓ = |y − f| + log 2",
    classic: { zh: "绝对值损失 → 第 03 章的稳健回归（中位数）", en: "Absolute loss → robust regression, the median (chapter 03)" }, y: { min: -2, max: 2, step: 0.1, init: 1 }, f: [-4, 4] },
  { id: "poisson", name: { zh: "泊松", en: "Poisson" }, q: { zh: "缺勤几次？", en: "How many absences?" }, head: "λ = eᶠ", loss: "ℓ = eᶠ − y·f + log y!",
    classic: { zh: "泊松损失 → 计数数据的广义线性模型", en: "Poisson loss → generalized linear models for counts" }, y: { min: 0, max: 6, step: 1, init: 3 }, f: [-3, 2.5] },
];

const TEXT = {
  title: { zh: "损失配方机：选一个输出分布，损失自己长出来", en: "The loss recipe machine: pick an output distribution, the loss follows" },
  y: { zh: "观测值 y", en: "Observed y" },
  best: { zh: "使损失最小的输出 f*", en: "Loss-minimising output f*" },
  mean: { zh: "此时预测的均值", en: "Predicted mean there" },
  head: { zh: "输出层", en: "Output head" },
  caption: {
    zh: "老师想对每个学生同时预测四件事：成绩、能否及格、等级、缺勤次数。同一个网络，只是最后一层的“头”和损失不同——而损失不必凭空设计：先选一个描述 y 的分布，让网络输出它的参数，损失就是负对数似然 −log Pr(y | f)。曲线是损失随网络原始输出 f 的变化，竖线是最小点。高斯给出平方损失，伯努利给出二元交叉熵，分类分布给出 softmax 交叉熵，拉普拉斯给出绝对值损失，泊松给出泊松损失。伯努利和分类分布在 y = 1 / y 为正确类时最小点在无穷远：f 越大越好，但每多一点好处越来越小。",
    en: "The teacher wants four predictions for every student: the score, pass or fail, the grade, and the number of absences. One network, with only the last-layer \"head\" and the loss differing. And the loss need not be invented: pick a distribution for y, let the network output its parameters, and the loss is the negative log-likelihood −log Pr(y | f). The curve shows the loss against the network's raw output f, and the vertical line marks its minimum. Gaussian gives the squared loss, Bernoulli binary cross-entropy, categorical softmax cross-entropy, Laplace the absolute loss, Poisson the Poisson loss. For Bernoulli and categorical with y = 1 (or the correct class) the minimum is at infinity: larger f is always better, by less and less.",
  },
};

export default function LossRecipe({ lang }: { lang: Lang }) {
  const [k, setK] = useState(0);
  const [ys, setYs] = useState(D.map((d) => d.y.init));
  const d = D[k], y = ys[k], dist = DISTS[d.id];
  const fStar = useMemo(() => {
    let best = d.f[0], bv = Infinity;
    for (let i = 0; i <= 2000; i++) { const f = d.f[0] + ((d.f[1] - d.f[0]) * i) / 2000; const v = dist.nll(y, f); if (v < bv - 1e-12) { bv = v; best = f; } }
    return best;
  }, [d, dist, y]);
  const atEdge = Math.abs(fStar - d.f[1]) < 1e-9 || Math.abs(fStar - d.f[0]) < 1e-9;
  const top = Math.max(dist.nll(y, d.f[0]), dist.nll(y, d.f[1]));
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {D.map((dd, i) => <button key={dd.id} className={`chip${k === i ? " on" : ""}`} onClick={() => setK(i)}>{dd.name[lang]}</button>)}
      </div>
      <div className="lr-card">
        <span className="lr-q">{d.q[lang]}</span>
        <span className="lr-row"><b>{TEXT.head[lang]}</b> {d.head}</span>
        <span className="lr-loss">{d.loss}</span>
        <span className="lr-classic">{d.classic[lang]}</span>
      </div>
      <CurvePlot xs={[]} ys={[]} xDomain={d.f} yDomain={[0, Math.min(12, top * 1.05)]} height={200}
        curves={[{ f: (f) => dist.nll(y, f), color: "var(--class-pos)", width: 3 }]}
        vlines={atEdge ? [] : [{ x: fStar, color: "var(--accent-2)" }]} xLabel="f" yLabel="ℓ" />
      <Slider label={TEXT.y[lang]} value={y} min={d.y.min} max={d.y.max} step={d.y.step} onChange={(v) => setYs(ys.map((o, i) => (i === k ? v : o)))} />
      <div className="w-stats">
        <Stat label={TEXT.best[lang]} value={atEdge ? (fStar > 0 ? "→ +∞" : "→ −∞") : fStar.toFixed(2)} tone="accent" />
        <Stat label={TEXT.mean[lang]} value={atEdge ? (fStar > 0 ? "→ 1" : "→ 0") : dist.mean(fStar).toFixed(2)} />
      </div>
    </Widget>
  );
}
