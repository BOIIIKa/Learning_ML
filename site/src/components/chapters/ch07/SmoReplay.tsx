import { useMemo } from "react";
import { blobs } from "../../../engines/data";
import { smoTrace } from "../../../engines/smo";
import type { Lang } from "../../../i18n/ui";
import TracePlayer from "../../viz/TracePlayer";
import Widget from "../../ui/Widget";

const TEXT = {
  title: { zh: "SMO 逐步回放：每次只动两个 α", en: "SMO step by step: two α's at a time" },
  dual: { zh: "对偶目标", en: "Dual objective" },
  caption: {
    zh: "每一步，黄色脉冲圈标出被选中的一对样本 (i, j)；它们的 α 沿约束直线更新后，超平面随之移动。下方曲线是对偶目标——它每一步都单调上升，直到 KKT 条件全部满足。",
    en: "At each step the pulsing yellow rings mark the chosen pair (i, j); their α's move along the constraint line and the hyperplane moves with them. The chart shows the dual objective — it rises monotonically at every step until all KKT conditions hold.",
  },
};

export default function SmoReplay({ lang }: { lang: Lang }) {
  const trace = useMemo(() => {
    const { X, y } = blobs(36, 21, 0.85);
    return smoTrace(X, y, 1);
  }, []);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="replay">
      <TracePlayer traces={[trace]} lang={lang} autoplay
        metrics={[{ key: "dual", label: TEXT.dual[lang], color: "var(--accent)" }]} />
    </Widget>
  );
}
