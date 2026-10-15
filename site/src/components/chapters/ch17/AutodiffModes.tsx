import { useMemo, useState } from "react";
import { jvp, testFn, vjp } from "../../../engines/backprop-init";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch17.css";

const TEXT = {
  title: { zh: "前向模式还是反向模式？", en: "Forward mode or reverse mode?" },
  n: { zh: "输入个数 n", en: "Inputs n" },
  m: { zh: "输出个数 m", en: "Outputs m" },
  passes: { zh: "已做的遍数", en: "Passes done" },
  fwd: { zh: "前向模式（每遍一列 Jv）", en: "Forward mode (a column Jv per pass)" },
  rev: { zh: "反向模式（每遍一行 uᵀJ）", en: "Reverse mode (a row uᵀJ per pass)" },
  needF: { zh: "前向模式需要的遍数", en: "Passes in forward mode" },
  needR: { zh: "反向模式需要的遍数", en: "Passes in reverse mode" },
  caption: {
    zh: "函数 f: Rⁿ → Rᵐ 的雅可比矩阵是 m × n。前向模式带着一个方向 v 和函数值一起往前算，一遍得到 Jv，也就是一列；反向模式先前向记下中间值，再从一个输出方向 u 往回算，一遍得到 uᵀJ，也就是一行。两种方法每遍的代价都只是一次函数计算的几倍。所以 n 小用前向，m 小用反向。训练网络时 m = 1（一个损失），n 是百万甚至万亿个参数：反向一遍就够，这就是反向传播。格子里的颜色是两种模式实际算出的导数，两者完全相同。",
    en: "The Jacobian of f: Rⁿ → Rᵐ is m × n. Forward mode carries a direction v along with the values and gets Jv, one column, per pass; reverse mode records the forward values, then works back from an output direction u and gets uᵀJ, one row, per pass. Each pass costs a small multiple of one function evaluation. So use forward mode when n is small, reverse mode when m is small. Training a network has m = 1 (one loss) and n in the millions or more: one reverse pass is enough, and that is backpropagation. The cell colours are the derivatives each mode actually computes; they agree exactly.",
  },
};

const f1 = (v: number) => v.toFixed(1);

export default function AutodiffModes({ lang }: { lang: Lang }) {
  const [n, setN] = useState(8);
  const [m, setM] = useState(2);
  const [rev, setRev] = useState(true);
  const [done, setDone] = useState(2);
  const A = useMemo(() => testFn(n, m), [n, m]);
  const x = useMemo(() => Array.from({ length: n }, (_, j) => ((j * 37) % 11) / 10 - 0.5), [n]);
  const need = rev ? m : n;
  const k = Math.min(done, need);
  // The computed part of J: k rows (reverse) or k columns (forward).
  const J = useMemo(() => {
    const out: (number | null)[][] = Array.from({ length: m }, () => new Array(n).fill(null));
    for (let p = 0; p < k; p++) {
      if (rev) vjp(A, x, Array.from({ length: m }, (_, i) => (i === p ? 1 : 0))).forEach((v, j) => (out[p][j] = v));
      else jvp(A, x, Array.from({ length: n }, (_, j) => (j === p ? 1 : 0))).forEach((v, i) => (out[i][p] = v));
    }
    return out;
  }, [A, x, k, rev, m, n]);
  const W = 460, cell = Math.min(40, (W - 20) / n, 200 / m), x0 = (W - cell * n) / 2;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <div className="chips">
        <button className={`chip${!rev ? " on" : ""}`} onClick={() => setRev(false)}>{TEXT.fwd[lang]}</button>
        <button className={`chip${rev ? " on" : ""}`} onClick={() => setRev(true)}>{TEXT.rev[lang]}</button>
      </div>
      <svg className="lm-svg" viewBox={`0 0 ${W} ${f1(cell * m + 16)}`} role="img">
        {J.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={f1(x0 + j * cell + 1)} y={f1(8 + i * cell + 1)} width={f1(cell - 2)} height={f1(cell - 2)}
            className={v === null ? "ad-cell" : v >= 0 ? "ad-cell pos" : "ad-cell neg"} opacity={v === null ? 1 : (0.25 + 0.75 * Math.min(1, Math.abs(v))).toFixed(2)} />
        )))}
      </svg>
      <Slider label={TEXT.n[lang]} value={n} min={1} max={12} step={1} onChange={setN} />
      <Slider label={TEXT.m[lang]} value={m} min={1} max={5} step={1} onChange={setM} />
      <Slider label={TEXT.passes[lang]} value={done} min={0} max={12} step={1} onChange={setDone} />
      <div className="w-stats">
        <Stat label={TEXT.needF[lang]} value={n} tone={!rev ? "accent" : undefined} />
        <Stat label={TEXT.needR[lang]} value={m} tone={rev ? "accent" : undefined} />
      </div>
    </Widget>
  );
}
