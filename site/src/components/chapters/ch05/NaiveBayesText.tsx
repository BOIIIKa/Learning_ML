import { useState } from "react";
import { nbLogScores, softmaxLog, wordProbs } from "../../../engines/generative";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

// The chapter's worked example: four-word messages to a clinic, labelled urgent / routine.
export const VOCAB = ["fever", "pain", "refill", "schedule"];
const VOCAB_ZH = ["发烧", "疼痛", "续药", "预约"];
const URGENT = [["fever", "pain", "fever", "pain"], ["fever", "pain", "fever", "schedule"]];
const ROUTINE = [["refill", "schedule", "schedule", "refill"], ["schedule", "refill", "schedule", "fever"], ["pain", "schedule", "schedule", "refill"]];
const count = (msgs: string[][]) => VOCAB.map((w) => msgs.flat().filter((v) => v === w).length);
const COUNTS = [count(URGENT), count(ROUTINE)];
const PRIORS = [URGENT.length / 5, ROUTINE.length / 5];

const TEXT = {
  title: { zh: "朴素贝叶斯分拣留言：拉普拉斯平滑", en: "Naive Bayes sorts messages: Laplace smoothing" },
  urgent: { zh: "紧急（2 条）", en: "Urgent (2 messages)" },
  routine: { zh: "常规（3 条）", en: "Routine (3 messages)" },
  alpha: { zh: "平滑强度 α", en: "Smoothing α" },
  word: { zh: "词", en: "word" },
  nU: { zh: "紧急计数", en: "N urgent" },
  nR: { zh: "常规计数", en: "N routine" },
  tU: { zh: "θ 紧急", en: "θ urgent" },
  tR: { zh: "θ 常规", en: "θ routine" },
  msg: { zh: "新留言", en: "New message" },
  add: { zh: "点词加入 →", en: "Click to add →" },
  empty: { zh: "（空：只剩先验）", en: "(empty: only the prior is left)" },
  clear: { zh: "清空", en: "Clear" },
  prior: { zh: "先验", en: "prior" },
  post: { zh: "P(紧急 | 留言)", en: "P(urgent | message)" },
  logodds: { zh: "对数几率（紧急 : 常规）", en: "Log-odds (urgent : routine)" },
  veto: { zh: "某个词在一类中从未出现 ⇒ 该类概率被直接判为 0", en: "A word never seen in one class ⇒ that class is vetoed outright" },
  caption: {
    zh: "上方是 5 条训练留言的词频，表中是估计出的 P(词 | 类)。下方条形图把对数几率拆成先验与每个词的贡献 log θ紧急,j − log θ常规,j：向右支持“紧急”，向左支持“常规”。把 α 拖到 0（极大似然估计）：“续药”从未在紧急留言中出现，于是一个词就把紧急的概率压成 0，不管其他证据多强。α = 1 就是拉普拉斯平滑，它相当于给每个词预先加了一次“虚拟计数”。",
    en: "Top: word counts in the 5 training messages, and the estimated P(word | class). Below, the log-odds is split into the prior and one contribution per word, log θ_urgent,j − log θ_routine,j: bars to the right favour “urgent”, to the left “routine”. Drag α to 0 (the maximum-likelihood estimate): “refill” never appeared in an urgent message, so that one word forces P(urgent) to 0 however strong the other evidence is. α = 1 is Laplace smoothing: one imaginary extra count for every word.",
  },
};

const BW = 460;
const ROW = 26;

export default function NaiveBayesText({ lang }: { lang: Lang }) {
  const [alpha, setAlpha] = useState(1);
  const [msg, setMsg] = useState<number[]>([0, 1, 2]);
  const word = (j: number) => (lang === "zh" ? `${VOCAB_ZH[j]} ${VOCAB[j]}` : VOCAB[j]);
  const theta = wordProbs(COUNTS, alpha);
  const x = VOCAB.map((_, j) => msg.filter((v) => v === j).length);
  const scores = nbLogScores(COUNTS, PRIORS, alpha, x);
  const post = softmaxLog(scores);
  const lo = scores[0] - scores[1];

  // One bar per message token plus the prior; −∞ contributions are drawn as a full-length red bar.
  const rows = [
    { label: TEXT.prior[lang], v: Math.log(PRIORS[0] / PRIORS[1]) },
    ...msg.map((j) => ({ label: word(j), v: Math.log(theta[0][j]) - Math.log(theta[1][j]) })),
  ];
  const finite = rows.map((r) => r.v).filter(Number.isFinite);
  const span = Math.max(1.5, ...finite.map(Math.abs));
  const mid = BW / 2 + 40;
  const half = BW - mid - 50;
  const bx = (v: number) => (Number.isFinite(v) ? (v / span) * half : v > 0 ? half : -half);
  const H = rows.length * ROW + 12;

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="nb5-train">
        {[URGENT, ROUTINE].map((msgs, k) => (
          <div key={k} className={`nb5-class ${k ? "neg" : "pos"}`}>
            <h4>{k ? TEXT.routine[lang] : TEXT.urgent[lang]}</h4>
            {msgs.map((m, i) => <div key={i} className="nb5-msg">“{m.join(" ")}”</div>)}
          </div>
        ))}
      </div>
      <div className="nb5-wrap">
        <table className="nb5-table">
          <thead>
            <tr><th>{TEXT.word[lang]}</th><th>{TEXT.nU[lang]}</th><th>{TEXT.nR[lang]}</th><th>{TEXT.tU[lang]}</th><th>{TEXT.tR[lang]}</th></tr>
          </thead>
          <tbody>
            {VOCAB.map((_, j) => (
              <tr key={j}>
                <td>{word(j)}</td>
                <td>{COUNTS[0][j]}</td>
                <td>{COUNTS[1][j]}</td>
                <td className={theta[0][j] === 0 ? "zero" : ""}>{theta[0][j].toFixed(3)}</td>
                <td className={theta[1][j] === 0 ? "zero" : ""}>{theta[1][j].toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Slider label={TEXT.alpha[lang]} value={alpha} min={0} max={3} step={0.05} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <div className="nb5-compose">
        <span className="nb5-label">{TEXT.msg[lang]}</span>
        {msg.length === 0 && <span className="nb5-empty">{TEXT.empty[lang]}</span>}
        {msg.map((j, i) => (
          <button key={i} className="nb5-token" title="×" onClick={() => setMsg(msg.filter((_, k) => k !== i))}>{VOCAB[j]} ×</button>
        ))}
      </div>
      <div className="nb5-compose">
        <span className="nb5-label">{TEXT.add[lang]}</span>
        <div className="chips">
          {VOCAB.map((w, j) => <button key={w} className="chip" disabled={msg.length >= 8} onClick={() => setMsg([...msg, j])}>+ {word(j)}</button>)}
          <button className="chip" onClick={() => setMsg([])}>{TEXT.clear[lang]}</button>
        </div>
      </div>
      <svg className="nb5-bars" viewBox={`0 0 ${BW} ${H}`} role="img">
        <line x1={mid} x2={mid} y1={2} y2={H - 4} className="nb5-zero" />
        {rows.map((r, i) => {
          const w = bx(r.v);
          const y = 6 + i * ROW;
          const cls = !Number.isFinite(r.v) ? "inf" : r.v > 0 ? "pos" : "neg";
          return (
            <g key={i}>
              <text x={8} y={y + 14} className="nb5-blab">{r.label}</text>
              <rect x={(w < 0 ? mid + w : mid).toFixed(1)} y={y + 2} width={Math.abs(w).toFixed(1)} height={ROW - 8} rx={4} className={`nb5-bar ${cls}`} />
              <text x={(w < 0 ? mid + w - 4 : mid + w + 4).toFixed(1)} y={y + 14} textAnchor={w < 0 ? "end" : "start"} className="nb5-bval">
                {Number.isFinite(r.v) ? (r.v >= 0 ? "+" : "") + r.v.toFixed(2) : r.v > 0 ? "+∞" : "−∞"}
              </text>
            </g>
          );
        })}
      </svg>
      <div>
        <div className="nb5-meter"><span style={{ width: `${(100 * (Number.isNaN(post[0]) ? 0.5 : post[0])).toFixed(1)}%` }} /></div>
        <div className="nb5-meter-lab"><span>{TEXT.urgent[lang].split(/[（(]/)[0]}</span><span>{TEXT.routine[lang].split(/[（(]/)[0]}</span></div>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.post[lang]} value={Number.isNaN(post[0]) ? "—" : post[0].toFixed(4)} tone="accent" />
        <Stat label={TEXT.logodds[lang]} value={Number.isFinite(lo) ? lo.toFixed(3) : lo > 0 ? "+∞" : "−∞"} tone={Number.isFinite(lo) ? undefined : "bad"} />
      </div>
      {!Number.isFinite(lo) && <p className="muted" style={{ margin: 0 }}>{TEXT.veto[lang]}</p>}
    </Widget>
  );
}
