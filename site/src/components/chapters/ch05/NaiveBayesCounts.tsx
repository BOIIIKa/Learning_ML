import { useState } from "react";
import { bernoulliPosterior, bernoulliProb } from "../../../engines/generative-classifiers";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "朴素贝叶斯：数一数，再平滑", en: "Naive Bayes: count, then smooth" },
  alpha: { zh: "拉普拉斯平滑 α", en: "Laplace smoothing α" },
  newp: { zh: "新病人", en: "New patient" },
  fever: { zh: "发热", en: "fever" },
  cough: { zh: "咳嗽", en: "cough" },
  pos: { zh: "需要复查（4 人）", en: "Follow-up (4)" },
  neg: { zh: "不需要（6 人）", en: "No follow-up (6)" },
  post: { zh: "P(复查 | 新病人)", en: "P(follow-up | patient)" },
  caption: {
    zh: "10 位确诊病人：需要复查的 4 人中 3 人发热、4 人全都咳嗽；不需要的 6 人中 1 人发热、2 人咳嗽。朴素贝叶斯假设给定类别后症状相互独立，于是只需数数。上面的条形是估计的条件概率 P(症状 | 类别)。选“既不发热也不咳嗽”的新病人并把 α 拖到 0：复查组从没出现过“不咳嗽”，概率是 0，一个零就把整个后验乘成 0。α = 1 的拉普拉斯平滑相当于每个格子先“假装”多见过一次，零就消失了。",
    en: "Ten diagnosed patients: of the 4 who needed a follow-up, 3 had a fever and all 4 a cough; of the 6 who did not, 1 had a fever and 2 a cough. Naive Bayes assumes symptoms are independent given the class, so all it does is count. The bars are the estimated conditional probabilities P(symptom | class). Pick the patient with neither fever nor cough and drag α to 0: the follow-up group never showed \"no cough\", so that probability is 0, and a single zero multiplies the whole posterior to 0. Laplace smoothing with α = 1 pretends every cell was seen once more, and the zero disappears.",
  },
};

const COUNTS: [number[], number[]] = [[3, 4], [1, 2]];
const NS: [number, number] = [4, 6];
const PATIENTS = [[0, 0], [1, 1], [1, 0], [0, 1]];
const W = 460;
const H = 190;
const f1 = (v: number) => v.toFixed(1);

export default function NaiveBayesCounts({ lang }: { lang: Lang }) {
  const [alpha, setAlpha] = useState(1);
  const [pi, setPi] = useState(0);
  const x = PATIENTS[pi];
  const post = bernoulliPosterior(COUNTS, NS, x, alpha);
  const label = (v: number[]) => `${v[0] ? "" : "¬"}${TEXT.fever[lang]} · ${v[1] ? "" : "¬"}${TEXT.cough[lang]}`;
  // Bars: P(feature takes the patient's value | class), for both classes and both features.
  const bars = [0, 1].flatMap((c) => [0, 1].map((j) => {
    const q = bernoulliProb(COUNTS[c][j], NS[c], alpha);
    return { c, j, v: x[j] ? q : 1 - q };
  }));
  const bw = 70;
  const sy = (v: number) => H - 30 - v * (H - 60);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <span className="nb-lab">{TEXT.newp[lang]}:</span>
        {PATIENTS.map((p, i) => (
          <button key={i} className={`chip${pi === i ? " on" : ""}`} onClick={() => setPi(i)}>{label(p)}</button>
        ))}
      </div>
      <svg className="lm-svg nb-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        {[0, 1].map((c) => <text key={c} x={c === 0 ? 95 : 300} y={16} textAnchor="middle" className={`nb-head ${c === 0 ? "pos" : "neg"}`}>{c === 0 ? TEXT.pos[lang] : TEXT.neg[lang]}</text>)}
        <line x1={20} x2={W - 20} y1={sy(0)} y2={sy(0)} className="cp-grid" />
        <line x1={20} x2={W - 20} y1={sy(1)} y2={sy(1)} className="cp-grid" />
        {bars.map((b, i) => {
          const x0 = (b.c === 0 ? 40 : 245) + b.j * (bw + 20);
          return (
            <g key={i}>
              <rect x={x0} y={f1(sy(b.v))} width={bw} height={f1(sy(0) - sy(b.v))} className={`nb-bar ${b.c === 0 ? "pos" : "neg"}${b.v === 0 ? " zero" : ""}`} />
              <text x={x0 + bw / 2} y={f1(sy(b.v) - 5)} textAnchor="middle" className="me-val">{b.v.toFixed(2)}</text>
              <text x={x0 + bw / 2} y={H - 12} textAnchor="middle" className="cp-tick">{`${x[b.j] ? "" : "¬"}${b.j === 0 ? TEXT.fever[lang] : TEXT.cough[lang]}`}</text>
            </g>
          );
        })}
      </svg>
      <Slider label={TEXT.alpha[lang]} value={alpha} min={0} max={3} step={0.1} onChange={setAlpha} format={(v) => v.toFixed(1)} />
      <div className="w-stats">
        <Stat label={TEXT.post[lang]} value={post.toFixed(3)} tone={post === 0 ? "bad" : "accent"} />
        <Stat label="P(+)" value="0.4" />
      </div>
    </Widget>
  );
}
