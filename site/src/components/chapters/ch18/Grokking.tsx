import { useState } from "react";
import { GROK, firstStep } from "../../../engines/dl-generalization";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch18.css";

const TEXT = {
  title: { zh: "顿悟：先背下来，很久以后才学会", en: "Grokking: memorise first, understand much later" },
  wd: { zh: "权重衰减", en: "Weight decay" },
  step: { zh: "训练步数（对数刻度）", en: "Training step (log scale)" },
  train: { zh: "训练准确率", en: "Training accuracy" },
  test: { zh: "测试准确率", en: "Test accuracy" },
  norm: { zh: "权重范数（相对初值）", en: "Weight norm (relative to init)" },
  now: { zh: "当前步数", en: "Step" },
  memo: { zh: "训练准确率到 99%", en: "Training accuracy hits 99%" },
  grok: { zh: "测试准确率到 95%", en: "Test accuracy hits 95%" },
  never: { zh: "未达到", en: "never" },
  caption: {
    zh: "任务是模 31 的加法 a + b mod 31：961 个算式里一半用来训练，另一半测试。网络是一层 256 个 ReLU 单元，输入是 a、b 的独热编码，全批量 AdamW 训练两万步（横轴是对数刻度的步数）。没有权重衰减时，不到一百步就背下了全部训练算式，测试准确率却一直接近 0，权重越长越大。加上权重衰减 1，训练准确率在第 69 步到 99%，测试准确率到两千多步才过半、约 9200 步到 95%，权重范数同时一路下降：先记住，再在正则的推动下找到更简单的解。衰减 0.1 太弱，两万步时测试才到 74%；衰减 3 太强，两万步时只有 63%。这些曲线在 Python 里预先训练好，在这里回放。",
    en: "The task is addition mod 31, a + b mod 31: half of the 961 sums are for training, the other half for testing. The network has one layer of 256 ReLU units on one-hot codes of a and b, trained with full-batch AdamW for twenty thousand steps (steps on a log scale across). Without weight decay it memorises every training sum in under a hundred steps, while the test accuracy stays near 0 and the weights keep growing. With weight decay 1, training accuracy reaches 99% at step 69, while test accuracy passes one half only after two thousand-odd steps and reaches 95% around step 9200, as the weight norm keeps falling: first memorise, then, pushed by the regulariser, find a simpler solution. A decay of 0.1 is too weak (74% test accuracy at twenty thousand steps); 3 is too strong (only 63% at twenty thousand steps). The curves were trained in Python beforehand and are replayed here.",
  },
};

const fmtStep = (s: number) => (s === Infinity ? "—" : s >= 1000 ? `${+(s / 1000).toFixed(1)}k` : `${s}`);

export default function Grokking({ lang }: { lang: Lang }) {
  const [w, setW] = useState(3);
  const run = GROK.runs[w];
  const [k, setK] = useState(run.step.length - 1);
  const n0 = run.norm[0];
  const memo = firstStep(run, "train", 0.99), grok = firstStep(run, "test", 0.95);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <MetricChart cursor={k} height={210} span={[0, 1]}
        series={[
          { key: "tr", label: TEXT.train[lang], values: run.train, color: "var(--accent-2)" },
          { key: "te", label: TEXT.test[lang], values: run.test, color: "var(--success)" },
          { key: "n", label: TEXT.norm[lang], values: run.norm.map((v) => v / n0), color: "var(--text-muted)", dashed: true },
        ]} />
      <Slider label={TEXT.wd[lang]} value={w} min={0} max={GROK.wds.length - 1} step={1} onChange={setW} format={(v) => `${GROK.wds[v]}`} />
      <Slider label={TEXT.step[lang]} value={k} min={0} max={run.step.length - 1} step={1} onChange={setK} format={(v) => fmtStep(run.step[v])} />
      <div className="w-stats">
        <Stat label={TEXT.memo[lang]} value={memo === Infinity ? TEXT.never[lang] : fmtStep(memo)} tone="accent" />
        <Stat label={TEXT.grok[lang]} value={grok === Infinity ? TEXT.never[lang] : fmtStep(grok)} tone={grok === Infinity ? "bad" : "good"} />
        <Stat label={`${TEXT.test[lang]} @ ${fmtStep(run.step[k])}`} value={run.test[k].toFixed(2)} />
      </div>
    </Widget>
  );
}
