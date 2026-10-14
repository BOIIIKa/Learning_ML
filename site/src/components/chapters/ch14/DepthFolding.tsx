import { useMemo, useState } from "react";
import { countPieces, deepTentParams, knotsFor, shallowFit, tentPow } from "../../../engines/neural-networks";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch14.css";

const TEXT = {
  title: { zh: "深度即折叠：同样的参数，深网络切出指数多段", en: "Depth as folding: same parameters, exponentially more pieces" },
  L: { zh: "层数 L（每层 2 个 ReLU）", en: "Layers L (2 ReLUs each)" },
  deep: { zh: "深网络的段数 2^L", en: "Deep net: pieces 2^L" },
  shallow: { zh: "同参数浅网络的段数（≤ 2L + 1）", en: "Shallow net, same parameters: pieces (≤ 2L + 1)" },
  params: { zh: "两者的参数个数", en: "Parameters of each" },
  layer: { zh: "第 {l} 层之后", en: "after layer {l}" },
  caption: {
    zh: "“帐篷函数” t(x) = 2·ReLU(x) − 4·ReLU(x − ½) 只用两个 ReLU，把区间 [0, 1] 对折一次：左半边拉伸成 [0, 1]，右半边翻转后也拉伸成 [0, 1]。把它叠 L 层，就对折了 L 次——上方的小图逐层显示，锯齿数每层翻一倍，共 2^L 段。下面的大图把它和参数个数完全相同的浅网络（一个隐层、2L 个单元，用最小二乘尽力拟合锯齿）放在一起：浅网络最多 2L + 1 段，L ≥ 3 后就跟不上了。深度的威力在于“重复利用”同一次折叠。",
    en: "The tent map t(x) = 2·ReLU(x) − 4·ReLU(x − ½) uses just two ReLUs to fold [0, 1] in half: the left half is stretched onto [0, 1], the right half is flipped and stretched onto [0, 1] too. Stacking it L times folds L times: the small plots above show it layer by layer, the teeth double every layer, 2^L pieces in all. The large plot sets it beside a shallow network with exactly the same number of parameters (one hidden layer of 2L units, least-squares fitted to the sawtooth as well as it can): the shallow net has at most 2L + 1 pieces and falls behind from L = 3 on. Depth wins by reusing the same fold again and again.",
  },
};

const W = 460, H = 180, P = 14;
const f1 = (v: number) => v.toFixed(1);
const path = (f: (x: number) => number, w: number, h: number, n = 400) =>
  Array.from({ length: n + 1 }, (_, i) => { const x = i / n; return `${i ? "L" : "M"}${f1(P + x * (w - 2 * P))},${f1(h - P - Math.max(-0.2, Math.min(1.2, f(x))) * (h - 2 * P))}`; }).join("");

export default function DepthFolding({ lang }: { lang: Lang }) {
  const [L, setL] = useState(3);
  const deep = (x: number) => tentPow(L, x);
  const shallow = useMemo(() => {
    const xs = Array.from({ length: 801 }, (_, i) => i / 800);
    return shallowFit(xs, xs.map((x) => tentPow(L, x)), knotsFor(2 * L, 0, 1)).f;
  }, [L]);
  const sp = useMemo(() => countPieces(shallow, 0, 1, 8192), [shallow]);
  const thumbs = Array.from({ length: L }, (_, i) => i + 1);
  const tw = Math.min(110, (W - 8 * (L - 1)) / L);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lm-svg df-svg" viewBox={`0 0 ${W} ${H + 92}`} role="img">
        {thumbs.map((l, i) => (
          <g key={l} transform={`translate(${f1(i * (tw + 8))},0)`}>
            <rect x={0} y={0} width={f1(tw)} height={70} className="df-frame" />
            <path d={path((x) => tentPow(l, x), tw, 70, 240)} className="df-thumb" />
            <text x={4} y={84} className="df-lab">{TEXT.layer[lang].replace("{l}", String(l))}</text>
          </g>
        ))}
        <g transform="translate(0,92)">
          <rect x={0} y={0} width={W} height={H} className="df-frame" />
          <path d={path(shallow, W, H, 600)} className="df-shallow" />
          <path d={path(deep, W, H, 1200)} className="df-deep" />
        </g>
      </svg>
      <Slider label={TEXT.L[lang]} value={L} min={1} max={6} step={1} onChange={setL} />
      <div className="w-stats">
        <Stat label={TEXT.deep[lang]} value={2 ** L} tone="accent" />
        <Stat label={TEXT.shallow[lang]} value={sp} />
        <Stat label={TEXT.params[lang]} value={deepTentParams(L)} />
      </div>
    </Widget>
  );
}
