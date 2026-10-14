import { useEffect, useRef, useState } from "react";

const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;

/**
 * Smoothly animate a numeric vector towards `target`. Returns the in-between value each frame.
 * Vectors of a different length (or undefined) jump instead of animating.
 */
export function useTween(target: number[] | undefined, ms = 450): number[] | undefined {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  const frame = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(frame.current);
    const from = current.current;
    const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!target || !from || from.length !== target.length || reduced || ms <= 0) {
      current.current = target;
      setValue(target);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const k = easeOutCubic(Math.min(1, (now - start) / ms));
      const next = target.map((v, i) => from[i] + (v - from[i]) * k);
      current.current = next;
      setValue(next);
      if (k < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [target, ms]);

  return value;
}
