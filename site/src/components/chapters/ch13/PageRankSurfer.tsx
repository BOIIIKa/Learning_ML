import { useMemo, useState } from "react";
import { pagerank, surfer } from "../../../engines/markov-chains";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch13.css";

const TEXT = {
  title: { zh: "随机冲浪者：PageRank 是平稳分布", en: "The random surfer: PageRank is a stationary distribution" },
  d: { zh: "阻尼系数 d", en: "Damping d" },
  steps: { zh: "冲浪步数（log₁₀）", en: "Surfer steps (log₁₀)" },
  top: { zh: "排名第一", en: "Top page" },
  gap: { zh: "访问频率与 PageRank 的最大差", en: "Max gap, visits vs PageRank" },
  caption: {
    zh: "六个网页和它们之间的链接。随机冲浪者以概率 d 沿当前页面的某条外链走，以概率 1 − d 跳到任意页面。圆的大小是幂迭代算出的 PageRank，环的大小是冲浪者实际的访问频率；步数越多，两者越吻合——PageRank 就是这条马尔可夫链的平稳分布。阻尼让链不可约、非周期，平稳分布唯一；d 越接近 1 越看重链接结构，也越慢收敛。A、B、C 三页就是讲义例题。",
    en: "Six web pages and the links between them. A random surfer follows a random out-link of the current page with probability d and jumps to any page with probability 1 − d. Disc size is the PageRank computed by power iteration; ring size is how often the surfer actually visited. With more steps the two agree: PageRank is the stationary distribution of this Markov chain. Damping makes the chain irreducible and aperiodic, so the stationary distribution is unique; the closer d is to 1, the more the link structure matters and the slower the convergence. Pages A, B and C are the worked example of the notes.",
  },
};

const LINKS = [[1, 2], [2], [0], [2, 4], [5, 3], [4]];
const POS: [number, number][] = [[120, 70], [70, 190], [190, 190], [300, 120], [360, 62], [420, 196]];
const NAMES = ["A", "B", "C", "D", "E", "F"];
const f1 = (v: number) => v.toFixed(1);

export default function PageRankSurfer({ lang }: { lang: Lang }) {
  const [d, setD] = useState(0.85);
  const [ls, setLs] = useState(2);
  const pr = useMemo(() => pagerank(LINKS, d), [d]);
  const vis = useMemo(() => surfer(LINKS, d, Math.round(10 ** ls), 11), [d, ls]);
  const gap = Math.max(...pr.map((v, i) => Math.abs(v - vis[i])));
  const top = pr.indexOf(Math.max(...pr));
  const R = (v: number) => 8 + 70 * Math.sqrt(v / 1.2);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lm-svg pg-svg" viewBox="0 0 460 260" role="img">
        <defs><marker id="pg-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" className="pg-head" /></marker></defs>
        {LINKS.flatMap((out, j) => out.map((i) => {
          const [x1, y1] = POS[j], [x2, y2] = POS[i];
          const L = Math.hypot(x2 - x1, y2 - y1);
          const ux = (x2 - x1) / L, uy = (y2 - y1) / L;
          const off = 4;
          return <line key={`${j}-${i}`} x1={f1(x1 + ux * R(pr[j]) - uy * off)} y1={f1(y1 + uy * R(pr[j]) + ux * off)} x2={f1(x2 - ux * (R(pr[i]) + 3) - uy * off)} y2={f1(y2 - uy * (R(pr[i]) + 3) + ux * off)} className="pg-edge" markerEnd="url(#pg-arrow)" />;
        }))}
        {POS.map(([x, y], i) => (
          <g key={i}>
            <circle cx={x} cy={y} r={f1(R(pr[i]))} className={`pg-node${i === top ? " top" : ""}`} />
            <circle cx={x} cy={y} r={f1(R(vis[i]))} className="pg-ring" />
            <text x={x} y={y + 4} textAnchor="middle" className="pg-lab">{NAMES[i]}</text>
            <text x={x} y={f1(y + R(pr[i]) + 14)} textAnchor="middle" className="pg-val">{pr[i].toFixed(3)}</text>
          </g>
        ))}
      </svg>
      <Slider label={TEXT.d[lang]} value={d} min={0.5} max={0.99} step={0.01} onChange={setD} format={(v) => v.toFixed(2)} />
      <Slider label={TEXT.steps[lang]} value={ls} min={1} max={5} step={0.1} onChange={setLs} format={(v) => `${Math.round(10 ** v)}`} />
      <div className="w-stats">
        <Stat label={TEXT.top[lang]} value={NAMES[top]} tone="accent" />
        <Stat label={TEXT.gap[lang]} value={gap.toFixed(3)} />
      </div>
    </Widget>
  );
}
