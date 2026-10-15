import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { LAND, ringLoss, type LandName } from "../../../engines/dl-generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch18.css";

const TEXT = {
  title: { zh: "损失地形切片", en: "Slices of the loss landscape" },
  skip: { zh: "20 层：无残差 vs 有残差", en: "20 layers: plain vs residual" },
  batch: { zh: "4 层：小批量 vs 全批量", en: "4 layers: small vs full batch" },
  plain: { zh: "无残差", en: "Plain" },
  residual: { zh: "残差", en: "Residual" },
  small: { zh: "小批量（10）", en: "Small batch (10)" },
  large: { zh: "全批量（200）", en: "Full batch (200)" },
  row: { zh: "一维切片的位置 β", en: "Position β of the 1-D slice" },
  slice: { zh: "沿 α 的一维切片：log₁₀ 损失", en: "1-D slice along α: log₁₀ loss" },
  edge: { zh: "距离 1 处的平均损失", en: "Mean loss at distance 1" },
  test: { zh: "测试准确率", en: "Test accuracy" },
  caption: {
    zh: "在双月数据上训练好网络，从训练得到的权重 θ* 出发，沿两个随机方向 δ、η 切出 L(θ* + αδ + βη)，α、β ∈ [−1, 1]。方向按“滤波器归一化”缩放：每个神经元的随机方向和它自己的权重一样长，否则权重大小不同的网络无法比较。颜色越浓损失越大（对数刻度），中心是训练得到的极小点。20 层的普通网络：离开中心一点，损失就涨到几十到几百；加上残差连接，同样的距离只有 2 到 3。小批量 SGD 停在比全批量训练更平的地方，可两者的测试准确率几乎一样（0.95 左右）：平坦与泛化的关系没有那么简单。数据在 Python 里预先算好（3 个随机种子结论相同）。",
    en: "Networks are trained on the two moons; starting from the trained weights θ*, we slice L(θ* + αδ + βη) along two random directions δ, η, with α, β ∈ [−1, 1]. The directions are filter-normalised: each neuron's random direction is scaled to the length of that neuron's own weights, otherwise networks with weights of different sizes could not be compared. The stronger the colour, the higher the loss (log scale); the centre is the trained minimum. For the plain 20-layer network the loss climbs to tens or hundreds as soon as you leave the centre; with skip connections it is only 2 to 3 at the same distance. Small-batch SGD ends somewhere flatter than full-batch training, yet both reach almost the same test accuracy (about 0.95): flatness and generalisation are not so simply related. The data were precomputed in Python (three random seeds agree).",
  },
};

const PAIRS: [LandName, LandName][] = [["plain", "residual"], ["small", "large"]];
const f1 = (v: number) => v.toFixed(1);
const LO = -2, HI = 2; // log₁₀ loss range of the shading
const shade = (v: number) => Math.min(1, Math.max(0, (Math.log10(Math.max(v, 1e-9)) - LO) / (HI - LO)));

function Heat({ S, label, row }: { S: number[][]; label: string; row: number }) {
  const K = S.length, cell = 160 / K;
  return (
    <div className="dg-heat">
      <div className="dg-sub">{label}</div>
      <svg className="lm-svg" viewBox="0 0 170 170" role="img">
        {S.map((r, j) => r.map((v, i) => (
          <rect key={`${i}-${j}`} x={f1(5 + i * cell)} y={f1(5 + (K - 1 - j) * cell)} width={f1(cell)} height={f1(cell)}
            className="dg-cell" fillOpacity={shade(v).toFixed(2)} />
        )))}
        <line x1={5} x2={165} y1={f1(5 + (K - 1 - row + 0.5) * cell)} y2={f1(5 + (K - 1 - row + 0.5) * cell)} className="dg-rowline" />
        <circle cx={85} cy={85} r={3} className="dg-star" />
      </svg>
    </div>
  );
}

export default function LossLandscape({ lang }: { lang: Lang }) {
  const [pair, setPair] = useState(0);
  const [row, setRow] = useState(12);
  const [a, b] = PAIRS[pair];
  const A = LAND.runs[a][0], B = LAND.runs[b][0];
  const K = LAND.K;
  const sx = scaleLinear().domain([-LAND.R, LAND.R]).range([0, K - 1]);
  const interp = (S: number[][]) => (x: number) => {
    const u = Math.min(K - 1, Math.max(0, sx(x))), i = Math.min(K - 2, Math.floor(u)), w = u - i;
    return Math.log10((1 - w) * S[row][i] + w * S[row][i + 1]);
  };
  const c = (K - 1) / 2;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <div className="chips">
        <button className={`chip${pair === 0 ? " on" : ""}`} onClick={() => setPair(0)}>{TEXT.skip[lang]}</button>
        <button className={`chip${pair === 1 ? " on" : ""}`} onClick={() => setPair(1)}>{TEXT.batch[lang]}</button>
      </div>
      <div className="dg-heats">
        <Heat S={A.loss} label={TEXT[a][lang]} row={row} />
        <Heat S={B.loss} label={TEXT[b][lang]} row={row} />
      </div>
      <div className="dg-sub">{TEXT.slice[lang]}</div>
      <CurvePlot xs={[]} ys={[]} xDomain={[-LAND.R, LAND.R]} yDomain={[-2.5, 3]} height={170}
        curves={[
          { f: interp(A.loss), color: "var(--danger)", width: 2.2, label: TEXT[a][lang] },
          { f: interp(B.loss), color: "var(--success)", width: 2.2, dashed: true, label: TEXT[b][lang] },
        ]} />
      <Slider label={TEXT.row[lang]} value={row} min={0} max={K - 1} step={1} onChange={setRow} format={(v) => (LAND.R * (v - c) / c).toFixed(2)} />
      <div className="w-stats">
        <Stat label={`${TEXT.edge[lang]} · ${TEXT[a][lang]}`} value={ringLoss(A.loss, c).toFixed(1)} tone="bad" />
        <Stat label={`${TEXT.edge[lang]} · ${TEXT[b][lang]}`} value={ringLoss(B.loss, c).toFixed(1)} tone="good" />
        <Stat label={TEXT.test[lang]} value={`${A.testAcc.toFixed(3)} / ${B.testAcc.toFixed(3)}`} />
      </div>
    </Widget>
  );
}
