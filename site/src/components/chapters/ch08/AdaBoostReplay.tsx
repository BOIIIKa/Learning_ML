import { useMemo } from "react";
import { adaboost } from "../../../engines/adaboost";
import { moons } from "../../../engines/data";
import type { Lang } from "../../../i18n/ui";
import TracePlayer from "../../viz/TracePlayer";
import Widget from "../../ui/Widget";

const TEXT = {
  title: { zh: "AdaBoost 逐轮回放", en: "AdaBoost, round by round" },
  error: { zh: "训练误差", en: "Training error" },
  bound: { zh: "上界 ∏Zₘ", en: "Bound ∏Zₘ" },
  caption: {
    zh: "每一轮加入一个“树桩”（只切一刀的树）。点的大小是样本权重：被分错的点（黄圈）下一轮会变大，迫使新树桩关注它们。背景是加权投票 ΣαₘGₘ(x) 的符号与强度；下方曲线中，训练误差始终在理论上界之下。",
    en: "Each round adds a stump (a one-cut tree). Point size is sample weight: misclassified points (yellow rings) grow for the next round, forcing the new stump to focus on them. The background shows the sign and strength of the weighted vote ΣαₘGₘ(x); below, training error always stays under the theoretical bound.",
  },
};

export default function AdaBoostReplay({ lang }: { lang: Lang }) {
  const trace = useMemo(() => {
    const { X, y } = moons(80, 4, 0.2);
    return adaboost(X, y, 30).trace;
  }, []);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="replay">
      <TracePlayer traces={[trace]} lang={lang} autoplay stepMs={700}
        metrics={[
          { key: "error", label: TEXT.error[lang], color: "var(--accent)" },
          { key: "bound", label: TEXT.bound[lang], color: "var(--accent-2)", dashed: true },
        ]} />
    </Widget>
  );
}
