import { useEffect, useMemo, useRef, useState } from "react";
import { maxRegions2d, patternGrid, randomUnits, regionsInWindow } from "../../../engines/neural-networks";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch14.css";

const TEXT = {
  title: { zh: "ReLU 网络的线性区域", en: "Linear regions of a ReLU network" },
  D: { zh: "隐单元个数 D", en: "Hidden units D" },
  reseed: { zh: "换一组权重", en: "New weights" },
  inside: { zh: "窗口内的区域数", en: "Regions in the window" },
  bound: { zh: "上界 1 + D + C(D,2)", en: "Bound 1 + D + C(D,2)" },
  caption: {
    zh: "两个输入的单隐层网络。每个 ReLU 单元在输入平面上画出一条直线 w·x + b = 0：一侧“开”，一侧“关”。D 条直线把平面切成多边形，每块多边形里开关组合固定，网络就是一个线性函数——同一种颜色就是同一个线性区域。D 条一般位置的直线最多切出 1 + D + C(D,2) 块：区域数随 D 平方增长，而参数只线性增长。换一组权重时，有的交点落在窗口外，窗口里看到的区域就会少于上界。",
    en: "A one-hidden-layer network on two inputs. Each ReLU unit draws a line w·x + b = 0 on the input plane: on one side it is on, on the other off. D lines cut the plane into polygons; inside each polygon the on/off pattern is fixed and the network is a single linear function, so one colour is one linear region. D lines in general position make at most 1 + D + C(D,2) pieces: the number of regions grows quadratically in D while the parameters grow only linearly. With new weights some crossings may fall outside the window, and fewer regions are visible than the bound.",
  },
};

const R = 2.5;
const N = 200;
const S = 300;
const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => ((x + R) / (2 * R)) * S;
const sy = (y: number) => ((R - y) / (2 * R)) * S;

export default function ReluRegions({ lang }: { lang: Lang }) {
  const [D, setD] = useState(3);
  const [seed, setSeed] = useState(12);
  const units = useMemo(() => randomUnits(D, seed), [D, seed]);
  const grid = useMemo(() => patternGrid(units, R, N), [units]);
  const inside = useMemo(() => regionsInWindow(units, R), [units]);
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvas.current!.getContext("2d")!;
    const img = ctx.createImageData(N, N);
    const css = getComputedStyle(canvas.current!);
    const base = [css.getPropertyValue("--accent-2"), css.getPropertyValue("--class-pos"), css.getPropertyValue("--accent"), css.getPropertyValue("--accent-3")];
    const parse = (c: string) => { const m = c.trim().match(/^#([0-9a-f]{6})$/i); const v = m ? parseInt(m[1], 16) : 0x58c4dd; return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
    const pal = base.map(parse);
    grid.cells.forEach((id, i) => {
      const c = pal[id % pal.length];
      const shade = 0.3 + 0.55 * (((id * 37) % 11) / 10);
      img.data.set([c[0], c[1], c[2], Math.round(255 * shade * 0.75)], i * 4);
    });
    ctx.putImageData(img, 0, 0);
  }, [grid]);
  // Each unit's boundary line, clipped to the square.
  const lines = units.map((u) => {
    const pts: [number, number][] = [];
    for (const x of [-R, R]) if (Math.abs(u.w[1]) > 1e-9) { const y = -(u.b + u.w[0] * x) / u.w[1]; if (Math.abs(y) <= R) pts.push([x, y]); }
    for (const y of [-R, R]) if (Math.abs(u.w[0]) > 1e-9) { const x = -(u.b + u.w[1] * y) / u.w[0]; if (Math.abs(x) <= R) pts.push([x, y]); }
    return pts.slice(0, 2);
  });
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="rr-box">
        <canvas ref={canvas} width={N} height={N} className="rr-canvas" />
        <svg className="rr-svg" viewBox={`0 0 ${S} ${S}`} role="img">
          {lines.map((p, i) => p.length === 2 && <line key={i} x1={f1(sx(p[0][0]))} y1={f1(sy(p[0][1]))} x2={f1(sx(p[1][0]))} y2={f1(sy(p[1][1]))} className="rr-line" />)}
        </svg>
      </div>
      <Slider label={TEXT.D[lang]} value={D} min={1} max={8} step={1} onChange={setD} />
      <div className="chips">
        <button className="chip" onClick={() => setSeed(seed + 1)}>{TEXT.reseed[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.inside[lang]} value={inside} tone="accent" />
        <Stat label={TEXT.bound[lang]} value={maxRegions2d(D)} />
      </div>
    </Widget>
  );
}
