import { useMemo, useState } from "react";
import { NEURON, neuronGraph, type NodeId } from "../../../engines/backprop-init";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider } from "../../ui/Widget";
import "./ch17.css";

const TEXT = {
  title: { zh: "计算图上的前向与反向", en: "Forward and backward on a computational graph" },
  stage: { zh: "播放到第几步", en: "Step" },
  w1: { zh: "权重 w₁", en: "Weight w₁" },
  fwd: { zh: "全部前向", en: "All forward" },
  bwd: { zh: "全部反向", en: "All backward" },
  reset: { zh: "重置", en: "Reset" },
  value: { zh: "值", en: "value" },
  grad: { zh: "梯度", en: "grad" },
  now: { zh: "当前：", en: "Now: " },
  start: { zh: "按步播放：先前向算值，再反向传梯度。", en: "Step through: values forward, then gradients backward." },
  caption: {
    zh: "一个神经元 L = ½(σ(w₁x₁ + w₂x₂ + b) − y)²，拆成乘、加、σ、减、平方五种基本运算。前向沿箭头算出每个节点的值（黄色）；反向从 ∂L/∂L = 1 出发，逆着箭头把梯度（蓝色）传回去：每个节点的梯度 = 下游节点的梯度 × 局部导数，多个下游时相加。走完一遍反向，所有参数的梯度都有了，代价和一次前向差不多。拖动 w₁，看所有值和梯度怎样跟着变。",
    en: "One neuron, L = ½(σ(w₁x₁ + w₂x₂ + b) − y)², split into five elementary operations: multiply, add, σ, subtract, square. The forward pass follows the arrows and computes every node's value (yellow); the backward pass starts from ∂L/∂L = 1 and sends gradients (blue) against the arrows: a node's gradient = its consumer's gradient × the local derivative, summed over consumers. One backward sweep yields the gradient of every parameter, at about the cost of one forward pass. Drag w₁ and watch every value and gradient follow.",
  },
};

// Node positions in a 460 × 250 canvas.
const POS: Record<NodeId, [number, number]> = {
  w1: [78, 30], x1: [78, 80], w2: [78, 140], x2: [78, 190], b: [78, 235], y: [330, 235],
  u1: [175, 55], u2: [175, 165], s: [255, 125], p: [325, 125], r: [390, 165], L: [438, 105],
};
const f1 = (v: number) => v.toFixed(1);

export default function BackpropGraph({ lang }: { lang: Lang }) {
  const [stage, setStage] = useState(0);
  const [w1, setW1] = useState(NEURON.w1);
  const g = useMemo(() => neuronGraph({ ...NEURON, w1 }), [w1]);
  const nF = g.forwardOrder.length, nB = g.backwardOrder.length;
  const shownV = new Set<NodeId>(["w1", "x1", "w2", "x2", "b", "y", ...g.forwardOrder.slice(0, Math.min(stage, nF))]);
  const shownG = new Set<NodeId>(g.backwardOrder.slice(0, Math.max(0, stage - nF)));
  if (stage >= nF + nB) for (const id of ["x1", "x2", "y"] as NodeId[]) shownG.add(id);
  const cur = stage === 0 ? null : stage <= nF ? g.forwardOrder[stage - 1] : g.backwardOrder[Math.min(stage - nF, nB) - 1];
  const byId = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
  const curNode = cur ? byId[cur] : null;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lm-svg bp-graph" viewBox="0 0 460 260" role="img">
        {g.nodes.flatMap((n) => n.inputs.map((src) => {
          const [x0, y0] = POS[src], [x1, y1] = POS[n.id];
          const back = shownG.has(n.id) && shownG.has(src);
          return <line key={`${src}-${n.id}`} x1={x0 + 16} y1={y0} x2={x1 - 18} y2={y1} className={`bp-edge${back ? " back" : ""}`} markerEnd="" />;
        }))}
        {g.nodes.map((n) => {
          const [x, y] = POS[n.id];
          const leaf = n.inputs.length === 0;
          return (
            <g key={n.id} className={`bp-node${cur === n.id ? " cur" : ""}${leaf ? " leaf" : ""}`}>
              <circle cx={x} cy={y} r={leaf ? 13 : 17} />
              <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" className="bp-op">{leaf ? n.label : n.op}</text>
              {!leaf && <text x={x} y={y - 22} textAnchor="middle" className="bp-name">{n.label}</text>}
              {shownV.has(n.id) && <text x={x + (leaf ? -18 : 0)} y={leaf ? y - 2 : y + 30} textAnchor={leaf ? "end" : "middle"} className="bp-val">{n.value.toFixed(3)}</text>}
              {shownG.has(n.id) && <text x={x + (leaf ? -18 : 0)} y={leaf ? y + 10 : y + 42} textAnchor={leaf ? "end" : "middle"} className="bp-grad">{n.grad.toFixed(4)}</text>}
            </g>
          );
        })}
      </svg>
      <div className="bp-now">
        {curNode ? <>{TEXT.now[lang]}<b>{curNode.label}</b>{stage <= nF ? ` ${TEXT.value[lang]} = ${curNode.value.toFixed(4)}` : ` ${TEXT.grad[lang]} ∂L/∂${curNode.label} = ${curNode.grad.toFixed(4)}`}{stage > nF ? `　${curNode.local}` : ""}</> : TEXT.start[lang]}
      </div>
      <Slider label={TEXT.stage[lang]} value={stage} min={0} max={nF + nB} step={1} onChange={setStage} />
      <Slider label={TEXT.w1[lang]} value={w1} min={-2} max={2} step={0.05} onChange={setW1} format={(v) => v.toFixed(2)} />
      <div className="chips">
        <button className="chip" onClick={() => setStage(nF)}>{TEXT.fwd[lang]}</button>
        <button className="chip" onClick={() => setStage(nF + nB)}>{TEXT.bwd[lang]}</button>
        <button className="chip" onClick={() => setStage(0)}>{TEXT.reset[lang]}</button>
      </div>
    </Widget>
  );
}
