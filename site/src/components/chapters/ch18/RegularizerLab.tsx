import { useMemo, useState } from "react";
import { LAB_STEPS, LEVELS, SNAP_EVERY, predict, students, trainLab, truthMse, type Reg } from "../../../engines/dl-generalization";
import { truth } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch18.css";

const TEXT = {
  title: { zh: "193 个参数，12 个学生", en: "193 parameters, 12 students" },
  wd: { zh: "权重衰减", en: "Weight decay" },
  dropout: { zh: "Dropout", en: "Dropout" },
  augment: { zh: "数据增强（学习时长加抖动）", en: "Augmentation (jitter the hours)" },
  strength: { zh: "强度", en: "Strength" },
  step: { zh: "训练步数", en: "Training step" },
  hours: { zh: "学习时长（小时）", en: "Study hours" },
  score: { zh: "分数", en: "Score" },
  train: { zh: "训练误差（12 人）", en: "Training error (12 students)" },
  test: { zh: "测试误差（200 名新学生）", en: "Test error (200 new students)" },
  trainS: { zh: "训练 MSE", en: "Train MSE" },
  testS: { zh: "测试 MSE", en: "Test MSE" },
  best: { zh: "最佳停止步（测试最低）", en: "Best stopping step (lowest test)" },
  floor: { zh: "真实曲线本身的测试 MSE", en: "Test MSE of the true curve itself" },
  caption: {
    zh: "老师用一个 1–64–1 的 ReLU 网络（193 个参数）拟合 12 个学生的学习时长和分数，Adam 训练 3000 步。不加正则时，网络把每个学生都记住了：训练 MSE 降到 2.4，200 名新学生上的测试 MSE 却从第 50 步的 62 涨到 184（真实曲线本身也有 55，那是噪声）。拖动“训练步数”看曲线怎样一步步扭曲；在测试误差最低处停下就是早停。再选一种正则、加大强度：权重衰减 3 把最终测试误差压到 61，Dropout 0.6 到 68，给学习时长加 1 小时的抖动到 64；但权重衰减 10 太强，网络连训练数据都拟合不了（287）。数据为合成示意。",
    en: "The teacher fits a 1–64–1 ReLU network (193 parameters) to 12 students' study hours and scores, with 3000 Adam steps. Without a regulariser the network memorises every student: the training MSE falls to 2.4, while the test MSE on 200 new students climbs from 62 at step 50 to 184 (the true curve itself scores 55: that is the noise). Drag the training step to watch the curve twist; stopping where the test error is lowest is early stopping. Then pick a regulariser and raise its strength: weight decay 3 brings the final test error down to 61, dropout 0.6 to 68, jittering the hours by 1 hour to 64; weight decay 10 is too strong and the network cannot even fit the training data (287). Synthetic illustrative data.",
  },
};

const REGS: Reg[] = ["wd", "dropout", "augment"];
const fmt = (reg: Reg, v: number) => (reg === "augment" ? `±${v} h` : `${v}`);

export default function RegularizerLab({ lang }: { lang: Lang }) {
  const [reg, setReg] = useState<Reg>("wd");
  const [level, setLevel] = useState(0);
  const [k, setK] = useState(LAB_STEPS / SNAP_EVERY);
  const run = useMemo(() => trainLab(reg, level), [reg, level]);
  const net = run.snaps[k];
  // Step 0 (the untrained network) is off the chart's scale; the curves start at step 50.
  const cursor = Math.max(k, 1) - 1;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {REGS.map((r) => (
          <button key={r} className={`chip${reg === r ? " on" : ""}`} onClick={() => setReg(r)}>{TEXT[r][lang]}</button>
        ))}
      </div>
      <div className="dg-two">
        <CurvePlot xs={students.xs} ys={students.ys} xDomain={[0, 10]} yDomain={[20, 100]} height={240}
          xLabel={TEXT.hours[lang]} yLabel={TEXT.score[lang]}
          curves={[
            { f: truth, color: "var(--text-muted)", dashed: true },
            { f: (x) => predict(net, x), color: "var(--accent)", width: 2.5 },
          ]} />
        <MetricChart cursor={cursor} height={200}
          series={[
            { key: "tr", label: TEXT.train[lang], values: run.train.slice(1), color: "var(--accent-2)" },
            { key: "te", label: TEXT.test[lang], values: run.test.slice(1), color: "var(--danger)" },
          ]} />
      </div>
      <Slider label={TEXT.strength[lang]} value={level} min={0} max={LEVELS[reg].length - 1} step={1} onChange={setLevel}
        format={(v) => fmt(reg, LEVELS[reg][v])} />
      <Slider label={TEXT.step[lang]} value={k} min={0} max={LAB_STEPS / SNAP_EVERY} step={1} onChange={setK}
        format={(v) => `${v * SNAP_EVERY}`} />
      <div className="w-stats">
        <Stat label={TEXT.trainS[lang]} value={run.train[k].toFixed(1)} />
        <Stat label={TEXT.testS[lang]} value={run.test[k].toFixed(1)} tone={run.test[k] > 100 ? "bad" : "accent"} />
        <Stat label={TEXT.best[lang]} value={`${run.best * SNAP_EVERY} · ${run.test[run.best].toFixed(1)}`} tone="good" />
        <Stat label={TEXT.floor[lang]} value={truthMse.toFixed(1)} />
      </div>
    </Widget>
  );
}
