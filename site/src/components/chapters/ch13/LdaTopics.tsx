import { useMemo, useState } from "react";
import { VOCAB, corpus, ldaGibbs, purity } from "../../../engines/markov-chains";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch13.css";

const TEXT = {
  title: { zh: "主题的涌现：吉布斯采样", en: "Topics emerging: Gibbs sampling" },
  s: { zh: "吉布斯扫描轮数", en: "Gibbs sweeps" },
  topic: { zh: "话题", en: "Topic" },
  docs: { zh: "每篇文档的话题比例", en: "Topic mix of each document" },
  purity: { zh: "话题纯度", en: "Topic purity" },
  caption: {
    zh: "12 篇短文档，用到 8 个词：前 4 个是医疗词，后 4 个是金融词；有的文档偏医疗，有的偏金融，有的各半。LDA 假设每篇文档是若干话题的混合，每个话题是词上的分布。开始时每个词的话题是随机分配的，两个话题都是大杂烩；吉布斯采样逐个词按“这篇文档里这个话题常见 × 这个话题里这个词常见”重新抽话题。几轮之后，两个话题自己分成了“医疗”和“金融”，文档的话题比例也和它们的真实构成对上了——全程没有任何标签。",
    en: "Twelve short documents over 8 words: the first 4 medical, the last 4 financial; some documents lean medical, some financial, some half and half. LDA assumes each document is a mix of topics and each topic a distribution over words. At first every word's topic is assigned at random and both topics are a jumble; Gibbs sampling then redraws each word's topic in proportion to \"this topic is common in this document × this word is common in this topic\". After a few sweeps the two topics have sorted themselves into \"medicine\" and \"finance\", and each document's topic mix matches its true make-up, all without any labels.",
  },
};

const f1 = (v: number) => v.toFixed(1);

export default function LdaTopics({ lang }: { lang: Lang }) {
  const docs = useMemo(() => corpus(), []);
  const states = useMemo(() => ldaGibbs(docs, 2, 8, 30, 5), [docs]);
  const [s, setS] = useState(0);
  const st = states[s];
  const V = VOCAB[lang];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="lda-row">
        {[0, 1].map((k) => {
          const tot = st.nkw[k].reduce((a, b) => a + b, 0) || 1;
          return (
            <svg key={k} className="lm-svg" viewBox="0 0 220 180" role="img">
              <text x={6} y={14} className={`lda-cap t${k}`}>{TEXT.topic[lang]} {k + 1}</text>
              {st.nkw[k].map((c, w) => {
                const h = Math.min(105, (370 * c) / tot);
                return (
                  <g key={w}>
                    <rect x={f1(10 + w * 25.5)} y={f1(125 - h)} width={20} height={f1(h)} className={`lda-bar ${w < 4 ? "med" : "fin"}`} />
                    <text x={f1(24 + w * 25.5)} y={136} textAnchor="end" transform={`rotate(-40 ${f1(24 + w * 25.5)} 136)`} className="lda-word">{V[w]}</text>
                  </g>
                );
              })}
            </svg>
          );
        })}
      </div>
      <div className="lda-docs">
        <span className="lda-docs-cap">{TEXT.docs[lang]}</span>
        {st.ndk.map((row, d) => {
          const p = row[0] / (row[0] + row[1]);
          return <span key={d} className="lda-doc"><span className="lda-doc-a" style={{ width: `${(100 * p).toFixed(0)}%` }} /></span>;
        })}
      </div>
      <Slider label={TEXT.s[lang]} value={s} min={0} max={states.length - 1} step={1} onChange={setS} />
      <div className="w-stats"><Stat label={TEXT.purity[lang]} value={`${(100 * purity(st)).toFixed(0)}%`} tone="accent" /></div>
    </Widget>
  );
}
