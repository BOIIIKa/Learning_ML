import { useMemo, useState } from "react";
import { lag1, logTarget, metropolis } from "../../../engines/markov-chains";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch13.css";

const TEXT = {
  title: { zh: "Metropolis–Hastings：一条会“随机游走”的链", en: "Metropolis–Hastings: a chain that wanders" },
  step: { zh: "提议步长（log₁₀）", en: "Proposal step (log₁₀)" },
  n: { zh: "已走的步数", en: "Steps taken" },
  acc: { zh: "接受率", en: "Acceptance rate" },
  ac: { zh: "相邻样本的相关", en: "Lag-1 autocorrelation" },
  caption: {
    zh: "背景是一个有两个峰的后验分布（只知道它正比于什么，不知道归一化常数）。链从左上角出发：每一步随机提议一个新位置，若密度变大就接受，变小则以密度之比的概率接受。走足够多步后，样本的分布就是目标分布。步长太小：几乎每步都接受，但挪得很慢，相邻样本高度相关，很久才到另一个峰；步长太大：提议总落在密度很低处，几乎全被拒绝，链原地不动。适中的步长（接受率二三成到五成）混合最快。",
    en: "The background is a two-peaked posterior known only up to its normalizing constant. The chain starts at the top left: each step proposes a random new position, accepts it if the density goes up, and accepts it with probability equal to the density ratio if it goes down. After enough steps the samples are distributed according to the target. With a tiny step almost every move is accepted but the chain crawls, consecutive samples are highly correlated, and it takes ages to reach the other peak; with a huge step proposals land where the density is tiny and are nearly all rejected, so the chain stands still. A moderate step (acceptance around 20–50%) mixes fastest.",
  },
};

const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => 230 + x * 62;
const sy = (y: number) => 150 - y * 52;
const N = 3000;

export default function MhSampler({ lang }: { lang: Lang }) {
  const [ls, setLs] = useState(-0.1);
  const [n, setN] = useState(N);
  const run = useMemo(() => metropolis(N, 10 ** ls, 7), [ls]);
  const heat = useMemo(() => {
    const cells: { x: number; y: number; v: number }[] = [];
    for (let i = 0; i < 37; i++) for (let j = 0; j < 29; j++) {
      const x = -3.7 + i * 0.2, y = -2.8 + j * 0.2;
      cells.push({ x, y, v: Math.exp(logTarget(x, y)) });
    }
    const m = Math.max(...cells.map((c) => c.v));
    return cells.map((c) => ({ ...c, v: c.v / m }));
  }, []);
  const shown = run.path.slice(0, n + 1);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <svg className="lm-svg mh-svg" viewBox="0 0 460 300" role="img">
        {heat.map((c, i) => <rect key={i} x={f1(sx(c.x - 0.1))} y={f1(sy(c.y + 0.1))} width={12.4} height={10.4} className="mh-heat" opacity={(0.85 * c.v).toFixed(2)} />)}
        <path d={"M" + shown.map((p) => `${f1(sx(p[0]))},${f1(sy(p[1]))}`).join("L")} className="mh-path" />
        <circle cx={f1(sx(shown.at(-1)![0]))} cy={f1(sy(shown.at(-1)![1]))} r={5} className="mh-head" />
      </svg>
      <Slider label={TEXT.step[lang]} value={ls} min={-1.5} max={1} step={0.05} onChange={setLs} format={(v) => (10 ** v).toPrecision(2)} />
      <Slider label={TEXT.n[lang]} value={n} min={0} max={N} step={10} onChange={setN} />
      <div className="w-stats">
        <Stat label={TEXT.acc[lang]} value={`${(100 * run.acc).toFixed(0)}%`} tone="accent" />
        <Stat label={TEXT.ac[lang]} value={lag1(run.path).toFixed(3)} />
      </div>
    </Widget>
  );
}
