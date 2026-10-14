import { useMemo, useState } from "react";
import { elbo1, loglik1, mStep1, oneD, resp1 } from "../../../engines/em-mixtures";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "EM 的几何：贴合下界，再把它推高", en: "The geometry of EM: touch the lower bound, then push it up" },
  t: { zh: "EM 迭代 t", en: "EM iteration t" },
  th: { zh: "当前 θ_t", en: "Current θ_t" },
  lp: { zh: "log p(x | θ_t)", en: "log p(x | θ_t)" },
  kl: { zh: "M 步后的间隙 = KL", en: "Gap after the M-step = KL" },
  x: { zh: "参数 θ", en: "parameter θ" },
  caption: {
    zh: "40 个数据点来自两个单位方差高斯的等权混合，一个中心在 0（已知），另一个在 θ（未知）。黄线是对数似然 log p(x | θ)。E 步在当前 θ_t 处取 q = 后验，得到的 ELBO（蓝线）处处在黄线下方，且恰好在 θ_t 处与它相切——此时 KL(q ‖ p(z | x)) = 0。M 步找 ELBO 的最高点 θ_{t+1}（竖虚线），它也必然让黄线更高：log p(θ_{t+1}) ≥ ELBO(θ_{t+1}) ≥ ELBO(θ_t) = log p(θ_t)。拖动 t 看“贴合–上推”一次次重复，直到停在局部极大。",
    en: "40 data points from an equal mixture of two unit-variance Gaussians, one centred at 0 (known), the other at θ (unknown). The yellow curve is the log-likelihood log p(x | θ). The E-step sets q to the posterior at the current θ_t, and the resulting ELBO (blue) lies below the yellow curve everywhere and touches it exactly at θ_t, where KL(q ‖ p(z | x)) = 0. The M-step moves to the ELBO's highest point θ_{t+1} (dashed vertical), which must also raise the yellow curve: log p(θ_{t+1}) ≥ ELBO(θ_{t+1}) ≥ ELBO(θ_t) = log p(θ_t). Drag t to watch touch-then-push repeat until it stops at a local maximum.",
  },
};

export default function ElboBound({ lang }: { lang: Lang }) {
  const xs = useMemo(() => oneD(), []);
  const ths = useMemo(() => {
    const out = [-0.6];
    for (let k = 0; k < 10; k++) out.push(mStep1(xs, resp1(xs, out.at(-1)!)));
    return out;
  }, [xs]);
  const [t, setT] = useState(0);
  const th = ths[t];
  const q = useMemo(() => resp1(xs, th), [xs, th]);
  const next = ths[t + 1] ?? th;
  const lo = Math.min(...[-1.5, 4].map((v) => loglik1(xs, v)));
  const hi = loglik1(xs, ths.at(-1)!);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={[th]} ys={[loglik1(xs, th)]} xDomain={[-1.5, 4]} yDomain={[lo - 5, hi + 4]} height={280} xLabel={TEXT.x[lang]}
        curves={[{ f: (v) => loglik1(xs, v), color: "var(--accent)" }, { f: (v) => elbo1(xs, q, v), color: "var(--accent-2)" }]}
        vlines={[{ x: th, color: "var(--ink)" }, { x: next, color: "var(--accent-2)" }]} />
      <Slider label={TEXT.t[lang]} value={t} min={0} max={ths.length - 2} step={1} onChange={setT} />
      <div className="w-stats">
        <Stat label={TEXT.th[lang]} value={th.toFixed(3)} tone="accent" />
        <Stat label={TEXT.lp[lang]} value={loglik1(xs, th).toFixed(2)} />
        <Stat label={TEXT.kl[lang]} value={(loglik1(xs, next) - elbo1(xs, q, next)).toFixed(3)} />
      </div>
    </Widget>
  );
}
