import { useMemo, useState } from "react";
import { SCHED_EPOCHS, SCHED_PEAKS, SCHEDULES, lrAt, trainSchedule, type Schedule } from "../../../engines/optimization";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Stat } from "../../ui/Widget";
import "./ch16.css";

const TEXT = {
  title: { zh: "学习率调度：同一个网络，四种节奏", en: "Learning-rate schedules: one network, four rhythms" },
  peak: { zh: "峰值学习率", en: "Peak learning rate" },
  lrs: { zh: "学习率（相对峰值）", en: "Learning rate (relative to peak)" },
  loss: { zh: "训练损失（每轮）", en: "Training loss (per epoch)" },
  final: { zh: "第 60 轮的损失", en: "Loss at epoch 60" },
  names: {
    constant: { zh: "常数", en: "Constant" }, step: { zh: "阶梯 ÷10", en: "Step ÷10" }, cosine: { zh: "余弦", en: "Cosine" }, warmup: { zh: "预热 + 余弦", en: "Warm-up + cosine" },
  } as Record<Schedule, Record<Lang, string>>,
  caption: {
    zh: "一个 2-16-16-1 的 tanh 网络在双月数据上做二分类，小批量 SGD（批大小 8，动量 0.9）训练 60 轮，四种调度只差在学习率怎样随时间变化。峰值 0.3 时，常数学习率一直在谷底附近乱跳，损失降不下去；阶梯和余弦先用大步走，后期把步子收小，才能落进谷底；先预热 6 轮再余弦，开头几轮的损失也不再猛跳。峰值只有 0.03 时，过早衰减反而拖慢训练：调度是用来驾驭大学习率的。",
    en: "A 2-16-16-1 tanh network classifies the two-moons data, trained for 60 epochs with minibatch SGD (batch 8, momentum 0.9); the four runs differ only in how the learning rate changes over time. At a peak of 0.3 a constant rate keeps bouncing around the valley floor and the loss will not come down; step and cosine schedules take big steps first and shrink them later, so they settle into the floor; warming up for 6 epochs before the cosine also stops the loss from jumping in the first epochs. At a peak of only 0.03, decaying early just slows training down: schedules are for taming large learning rates.",
  },
};

const COLORS: Record<Schedule, string> = { constant: "var(--text-muted)", step: "var(--accent-2)", cosine: "var(--success)", warmup: "var(--accent)" };

export default function LrSchedules({ lang }: { lang: Lang }) {
  const [peak, setPeak] = useState(0.3);
  const runs = useMemo(() => Object.fromEntries(SCHEDULES.map((s) => [s, trainSchedule(s, peak)])) as Record<Schedule, number[]>, [peak]);
  const T = SCHED_EPOCHS;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <div className="chips">
        <span className="op-chiplab">{TEXT.peak[lang]}</span>
        {SCHED_PEAKS.map((p) => <button key={p} className={`chip${peak === p ? " on" : ""}`} onClick={() => setPeak(p)}>{p}</button>)}
      </div>
      <div className="op-two">
        <div>
          <div className="op-sub">{TEXT.lrs[lang]}</div>
          <MetricChart cursor={T} height={150}
            series={SCHEDULES.map((s) => ({ key: s, label: TEXT.names[s][lang], color: COLORS[s], values: Array.from({ length: T + 1 }, (_, e) => lrAt(s, 1, e, T)) }))} />
        </div>
        <div>
          <div className="op-sub">{TEXT.loss[lang]}</div>
          <MetricChart cursor={T} height={150}
            series={SCHEDULES.map((s) => ({ key: s, label: TEXT.names[s][lang], color: COLORS[s], values: runs[s].map((v) => Math.min(v, 1.5)) }))} />
        </div>
      </div>
      <div className="w-stats">
        {SCHEDULES.map((s) => <Stat key={s} label={`${TEXT.names[s][lang]} · ${TEXT.final[lang]}`} value={runs[s][T].toFixed(3)} tone={s === "warmup" ? "accent" : undefined} />)}
      </div>
    </Widget>
  );
}
