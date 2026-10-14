import { useMemo, useState } from "react";
import { blobs, rng } from "../../../engines/data";
import { smo } from "../../../engines/smo";
import type { Lang } from "../../../i18n/ui";
import Plot2D, { type ExtraLine } from "../../viz/Plot2D";
import Widget, { Stat } from "../../ui/Widget";

type Line = { w: [number, number]; b: number };

const TEXT = {
  title: { zh: "情景：哪条线更值得信任？", en: "Scenario: which line would you trust?" },
  a: { zh: "直线 A", en: "Line A" },
  b: { zh: "直线 B", en: "Line B" },
  svm: { zh: "最大间隔", en: "Max margin" },
  newPatients: { zh: "来了 40 位新病人", en: "40 new patients arrive" },
  reveal: { zh: "显示最大间隔直线", en: "Reveal the max-margin line" },
  reset: { zh: "重置", en: "Reset" },
  margin: { zh: "间隔", en: "margin" },
  errors: { zh: "新病人误判", en: "new mistakes" },
  seen: { zh: "已到新病人", en: "new patients" },
  caption: {
    zh: "三条直线在训练数据上都 0 错误。点按钮让新病人（空心点）按同一分布到来，看看谁犯错更少。点选上方的直线名称，可高亮它误判的病人（红色）。数据为合成示意。",
    en: "All three lines make zero mistakes on the training data. Bring in new patients (hollow points) from the same distribution and see who errs less. Click a line's name to highlight the patients it misclassifies (red). Synthetic illustrative data.",
  },
};

const COLORS = ["#ffb547", "#b388ff", "#4fd1ff"];

const geometricMargin = (l: Line, X: number[][], y: number[]) =>
  Math.min(...X.map((x, i) => (y[i] * (l.w[0] * x[0] + l.w[1] * x[1] + l.b)) / Math.hypot(...l.w)));

/** Two separating lines that hug the data: smallest positive margins with clearly different angles. */
function huggingLines(X: number[][], y: number[]): [Line, Line] {
  const rand = rng(11);
  const found: { line: Line; margin: number; angle: number }[] = [];
  for (let k = 0; k < 4000; k++) {
    const angle = Math.PI * rand();
    const w: [number, number] = [Math.cos(angle), Math.sin(angle)];
    const line = { w, b: -(w[0] * (2 * rand() - 1) * 1.2 + w[1] * (2 * rand() - 1) * 1.2) };
    for (const sign of [1, -1]) {
      const l = { w: [sign * w[0], sign * w[1]] as [number, number], b: sign * line.b };
      const m = geometricMargin(l, X, y);
      if (m > 0.04) found.push({ line: l, margin: m, angle });
    }
  }
  found.sort((p, q) => p.margin - q.margin);
  const first = found[0];
  const second = found.find((f) => Math.abs(f.angle - first.angle) > 0.5) ?? found[1];
  return [first.line, second.line];
}

export default function ScenarioLines({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => blobs(30, 3, 0.55), []);
  const lines = useMemo(() => {
    const [a, b] = huggingLines(X, y);
    const r = smo(X, y, { C: 1000, maxSteps: 3000 });
    return [a, b, { w: r.w, b: r.b }];
  }, [X, y]);
  const [revealed, setRevealed] = useState(false);
  const [selected, setSelected] = useState(0);
  const [round, setRound] = useState(0);

  const test = useMemo(() => {
    const out = { X: [] as number[][], y: [] as number[] };
    for (let r = 1; r <= round; r++) {
      const d = blobs(40, 100 + r, 0.55);
      out.X.push(...d.X);
      out.y.push(...d.y);
    }
    return out;
  }, [round]);

  const wrong = (l: Line) => test.X.map((x, i) => test.y[i] * (l.w[0] * x[0] + l.w[1] * x[1] + l.b) < 0);
  const visible = revealed ? [0, 1, 2] : [0, 1];
  const names = [TEXT.a[lang], TEXT.b[lang], TEXT.svm[lang]];
  const extraLines: ExtraLine[] = visible.map((i) => ({ ...lines[i], color: COLORS[i], dashed: i !== selected }));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {visible.map((i) => (
          <button key={i} className={`chip${i === selected ? " on" : ""}`} onClick={() => setSelected(i)}>
            <i style={{ background: COLORS[i] }} /> {names[i]}
          </button>
        ))}
      </div>
      <Plot2D X={X} y={y} extraLines={extraLines} extraPoints={{ ...test, bad: wrong(lines[selected]) }} showMargin={false}
        domain={{ x: [-3.6, 3.6], y: [-3.2, 3.4] }} />
      <div className="w-stats">
        {visible.map((i) => {
          const errs = wrong(lines[i]).filter(Boolean).length;
          return (
            <Stat key={i} label={`${names[i]} · ${TEXT.errors[lang]} · ${TEXT.margin[lang]} ${geometricMargin(lines[i], X, y).toFixed(2)}`}
              value={errs} tone={round === 0 ? undefined : errs === 0 ? "good" : "bad"} />
          );
        })}
        <Stat label={TEXT.seen[lang]} value={test.X.length} tone="accent" />
      </div>
      <div className="w-row">
        <button className="btn btn-primary" onClick={() => setRound(round + 1)}>{TEXT.newPatients[lang]}</button>
        {!revealed && <button className="btn" onClick={() => { setRevealed(true); setSelected(2); }}>{TEXT.reveal[lang]}</button>}
        {round > 0 && <button className="btn" onClick={() => setRound(0)}>{TEXT.reset[lang]}</button>}
      </div>
    </Widget>
  );
}
