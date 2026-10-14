import { useEffect, useId, useMemo, useRef, useState } from "react";
import { r6 } from "../../../engines/data";
import { bfsOrder, buildKd, dist, kdPoints, kdSearch, subtree, type KdEvent } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch06.css";

const TEXT = {
  title: { zh: "kd 树：逐层切分与最近邻搜索", en: "kd-tree: recursive splits and nearest-neighbour search" },
  build: { zh: "▶ 建树", en: "▶ Build" },
  search: { zh: "▶ 搜索", en: "▶ Search" },
  buildStep: { zh: "建树步数", en: "Build step" },
  searchStep: { zh: "搜索步数", en: "Search step" },
  computed: { zh: "计算的距离", en: "Distances computed" },
  pruned: { zh: "被剪掉的点", en: "Points pruned" },
  best: { zh: "当前最近距离 r", en: "Best distance r" },
  hint: { zh: "点击左图任意位置放置查询点 q", en: "Click anywhere on the left to place a query q" },
  caption: {
    zh: "建树：每一层在当前格子里取中位数切一刀，深度为偶数时切 x（竖线），奇数时切 y（横线），所以树是平衡的、深度约 log₂n。搜索：先沿 q 所在的一侧一路走到底，得到一个候选最近点和半径 r；回溯时，只有当分割线到 q 的距离小于 r（圆越过了分割线）时，另一侧才可能藏着更近的点，否则整棵子树被剪掉（灰色）。",
    en: "Building: each level cuts the current cell at the median, on x (vertical line) at even depths and on y (horizontal line) at odd depths, so the tree is balanced with depth about log₂n. Searching: first walk down the side that contains q to get a candidate and a radius r; on the way back, the other side can only hide a closer point if the splitting line is nearer to q than r (the circle crosses the line) — otherwise that whole subtree is pruned (grey).",
  },
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const S = 300; // plane size in px
const PAD = 10;
const px = (v: number) => r6(PAD + (v / 10) * S);
const py = (v: number) => r6(PAD + S - (v / 10) * S);
const f1 = (v: number) => v.toFixed(1);

function noteFor(ev: KdEvent, P: number[][], nodes: ReturnType<typeof buildKd>, q: number[], lang: Lang): string {
  const n = nodes[ev.node];
  const L = LETTERS[n.point];
  const r = ev.r.toFixed(2);
  const gap = ev.gap?.toFixed(2);
  switch (ev.kind) {
    case "visit":
      return lang === "zh" ? `访问 ${L}：距离 ${dist(P[n.point], q, "l2").toFixed(2)}` : `Visit ${L}: distance ${dist(P[n.point], q, "l2").toFixed(2)}`;
    case "update":
      return lang === "zh" ? `${L} 更近 → 当前最近点 = ${L}，r = ${r}` : `${L} is closer → best = ${L}, r = ${r}`;
    case "cross": {
      const p = LETTERS[n.point];
      return lang === "zh" ? `回溯到 ${p}：分割线距 q 仅 ${gap} < r = ${r}，圆越过了分割线，必须搜另一侧`
        : `Back at ${p}: the split is only ${gap} < r = ${r} from q, the circle crosses it, so search the other side`;
    }
    case "prune": {
      const parent = LETTERS[nodes[n.parent!].point];
      const m = subtree(nodes, ev.node).length;
      return lang === "zh" ? `回溯到 ${parent}：分割线距 q 为 ${gap} ≥ r = ${r}，另一侧整棵子树（${m} 个点）被剪掉`
        : `Back at ${parent}: the split is ${gap} ≥ r = ${r} from q, so the whole subtree on the other side (${m} points) is pruned`;
    }
    default:
      return lang === "zh" ? `结束：最近邻是 ${LETTERS[ev.best]}，只算了 ${ev.visited.length} / ${P.length} 个距离`
        : `Done: the nearest neighbour is ${LETTERS[ev.best]}, after only ${ev.visited.length} / ${P.length} distances`;
  }
}

export default function KdTreeSearch({ lang }: { lang: Lang }) {
  const P = useMemo(() => kdPoints(), []);
  const nodes = useMemo(() => buildKd(P, [0, 10, 0, 10]), [P]);
  const order = useMemo(() => bfsOrder(nodes), [nodes]);
  const [built, setBuilt] = useState(nodes.length);
  const [q, setQ] = useState<[number, number]>([7.5, 2.5]);
  const search = useMemo(() => kdSearch(nodes, P, q), [nodes, P, q]);
  const [e, setE] = useState(search.events.length - 1);
  const [playing, setPlaying] = useState<"build" | "search" | null>(null);
  const plane = useRef<SVGSVGElement>(null);
  const clip = `kd-clip-${useId().replace(/:/g, "")}`;

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => {
      if (playing === "build") {
        if (built >= nodes.length) setPlaying(null);
        else setBuilt(built + 1);
      } else if (e >= search.events.length - 1) setPlaying(null);
      else setE(e + 1);
    }, playing === "build" ? 550 : 900);
    return () => clearTimeout(id);
  }, [playing, built, e, nodes.length, search.events.length]);

  const shown = new Set(order.slice(0, built));
  const isBuilt = built >= nodes.length;
  const ev = isBuilt ? search.events[Math.min(e, search.events.length - 1)] : null;
  const visited = new Set(ev?.visited ?? []);
  const prunedNodes = new Set((ev?.pruned ?? []).flatMap((id) => subtree(nodes, id)));
  const current = ev && ev.kind !== "done" ? ev.node : -1;
  const building = !isBuilt && built > 0 ? order[built - 1] : -1;

  const place = (evt: React.PointerEvent) => {
    const pt = new DOMPoint(evt.clientX, evt.clientY).matrixTransform(plane.current!.getScreenCTM()!.inverse());
    const x = Math.round(Math.max(0, Math.min(10, ((pt.x - PAD) / S) * 10)) * 10) / 10;
    const y = Math.round(Math.max(0, Math.min(10, ((S + PAD - pt.y) / S) * 10)) * 10) / 10;
    setQ([x, y]);
    setBuilt(nodes.length);
    setE(0);
    setPlaying("search");
  };

  // Tree layout: x by in-order rank, y by depth.
  const rank = new Map<number, number>();
  const inorder = (id: number | null) => {
    if (id === null) return;
    inorder(nodes[id].left);
    rank.set(id, rank.size);
    inorder(nodes[id].right);
  };
  inorder(0);
  const tx = (id: number) => r6(16 + (rank.get(id)! * 288) / Math.max(1, nodes.length - 1));
  const ty = (id: number) => r6(22 + nodes[id].depth * 52);

  const splitLine = (id: number, cls: string) => {
    const n = nodes[id];
    const s = P[n.point][n.axis];
    const [x0, x1, y0, y1] = n.rect;
    return n.axis === 0
      ? <line key={`s${id}`} x1={px(s)} x2={px(s)} y1={py(y0)} y2={py(y1)} className={cls} />
      : <line key={`s${id}`} x1={px(x0)} x2={px(x1)} y1={py(s)} y2={py(s)} className={cls} />;
  };

  const gapLine = (() => {
    if (!ev || (ev.kind !== "cross" && ev.kind !== "prune")) return null;
    const parentId = ev.kind === "cross" ? ev.node : nodes[ev.node].parent!;
    const n = nodes[parentId];
    const s = P[n.point][n.axis];
    const cls = `kd-gap ${ev.kind}`;
    return n.axis === 0
      ? <line x1={px(q[0])} x2={px(s)} y1={py(q[1])} y2={py(q[1])} className={cls} />
      : <line x1={px(q[0])} x2={px(q[0])} y1={py(q[1])} y2={py(s)} className={cls} />;
  })();
  const r = ev && Number.isFinite(ev.r) ? ev.r : 0;

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="split-2">
        <svg ref={plane} className="kd-plane" viewBox={`0 0 ${S + 2 * PAD} ${S + 2 * PAD}`} onPointerDown={place} role="img">
          <defs><clipPath id={clip}><rect x={PAD} y={PAD} width={S} height={S} /></clipPath></defs>
          <rect x={PAD} y={PAD} width={S} height={S} rx={6} className="plot-bg" />
          {building >= 0 && (() => {
            const [x0, x1, y0, y1] = nodes[building].rect;
            return <rect x={px(x0)} y={py(y1)} width={r6(px(x1) - px(x0))} height={r6(py(y0) - py(y1))} className="kd-cell" />;
          })()}
          {ev?.pruned.map((id) => {
            const [x0, x1, y0, y1] = nodes[id].rect;
            return <rect key={`p${id}`} x={px(x0)} y={py(y1)} width={r6(px(x1) - px(x0))} height={r6(py(y0) - py(y1))} className="kd-pruned-cell" />;
          })}
          {order.filter((id) => shown.has(id)).map((id) => splitLine(id, `kd-split ax${nodes[id].axis}${id === current || id === building ? " hot" : ""}`))}
          {ev && (
            <g clipPath={`url(#${clip})`}>
              {r > 0 && <circle cx={px(q[0])} cy={py(q[1])} r={f1((r / 10) * S)} className="kd-ball" />}
              {ev.best >= 0 && <line x1={px(q[0])} y1={py(q[1])} x2={px(P[ev.best][0])} y2={py(P[ev.best][1])} className="kd-bestline" />}
              {gapLine}
            </g>
          )}
          {nodes.map((n) => {
            const cls = !isBuilt ? (shown.has(n.id) ? "on" : "")
              : prunedNodes.has(n.id) ? "pruned" : n.point === ev?.best ? "best" : visited.has(n.id) ? "visited" : "on";
            return (
              <g key={n.id} transform={`translate(${px(P[n.point][0])},${py(P[n.point][1])})`}>
                {n.id === current && <circle r={10} className="kd-cur" />}
                <circle r={5} className={`kd-pt ${cls}`} />
                <text x={7} y={-6} className="kd-lab">{LETTERS[n.point]}</text>
              </g>
            );
          })}
          {ev && (
            <g transform={`translate(${px(q[0])},${py(q[1])})`}>
              <path d="M0,-8 L2.4,-2.4 L8,0 L2.4,2.4 L0,8 L-2.4,2.4 L-8,0 L-2.4,-2.4Z" className="kd-q" />
              <text x={8} y={14} className="kd-lab q">q</text>
            </g>
          )}
        </svg>
        <div className="kd-side">
          <svg className="kd-tree" viewBox="0 0 320 212" role="img">
            {nodes.map((n) => n.parent !== null && shown.has(n.id) && (
              <line key={`e${n.id}`} x1={tx(n.parent)} y1={ty(n.parent)} x2={tx(n.id)} y2={ty(n.id)} className={`kd-edge${prunedNodes.has(n.id) ? " pruned" : ""}`} />
            ))}
            {nodes.filter((n) => shown.has(n.id)).map((n) => {
              const cls = !isBuilt ? "on" : prunedNodes.has(n.id) ? "pruned" : n.point === ev?.best ? "best" : visited.has(n.id) ? "visited" : "on";
              return (
                <g key={n.id} transform={`translate(${tx(n.id)},${ty(n.id)})`} className="kd-tnode">
                  {(n.id === current || n.id === building) && <circle r={14} className="kd-cur" />}
                  <circle r={10} className={`kd-pt ${cls}`} />
                  <text y={4} textAnchor="middle" className="kd-tlab">{LETTERS[n.point]}</text>
                  <text y={22} textAnchor="middle" className={`kd-split-lab ax${n.axis}`}>{n.axis === 0 ? "x" : "y"}={P[n.point][n.axis]}</text>
                </g>
              );
            })}
          </svg>
          <p className="kd-note" aria-live="polite">
            {ev ? noteFor(ev, P, nodes, q, lang) : lang === "zh" ? `第 ${built} 刀` : `Cut ${built}`}
          </p>
          <p className="kd-hint">{TEXT.hint[lang]}</p>
        </div>
      </div>
      <div className="w-row">
        <button className="btn" onClick={() => { setBuilt(0); setPlaying("build"); }}>{TEXT.build[lang]}</button>
        <Slider label={TEXT.buildStep[lang]} value={built} min={0} max={nodes.length} step={1} onChange={(v) => { setPlaying(null); setBuilt(v); }} format={(v) => `${v}/${nodes.length}`} />
      </div>
      <div className="w-row">
        <button className="btn btn-primary" onClick={() => { setBuilt(nodes.length); setE(0); setPlaying("search"); }}>{TEXT.search[lang]}</button>
        <Slider label={TEXT.searchStep[lang]} value={Math.min(e, search.events.length - 1)} min={0} max={search.events.length - 1} step={1}
          onChange={(v) => { setPlaying(null); setBuilt(nodes.length); setE(v); }} format={(v) => `${v + 1}/${search.events.length}`} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.computed[lang]} value={ev ? `${ev.visited.length} / ${P.length}` : "–"} tone="accent" />
        <Stat label={TEXT.pruned[lang]} value={ev ? String(prunedNodes.size) : "–"} tone="good" />
        <Stat label={TEXT.best[lang]} value={ev && Number.isFinite(ev.r) ? ev.r.toFixed(2) : "–"} />
      </div>
    </Widget>
  );
}
