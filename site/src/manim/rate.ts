// Rate functions, following manim's definitions (manim/utils/rate_functions.py).
export type Rate = (t: number) => number;

export const linear: Rate = (t) => t;

/** manim's default: a smooth S-curve with zero velocity at both ends (inflection 10). */
export const smooth: Rate = (t) => {
  const s = 1 - t;
  return t ** 3 * (10 * s * s + 5 * s * t + t * t);
};

export const rushInto: Rate = (t) => 2 * smooth(t / 2);
export const rushFrom: Rate = (t) => 2 * smooth(t / 2 + 0.5) - 1;
export const thereAndBack: Rate = (t) => smooth(t < 0.5 ? 2 * t : 2 * (1 - t));

/** CSS easing strings that approximate the rate functions for the Web Animations API. */
export const EASE = {
  smooth: "cubic-bezier(0.45, 0, 0.25, 1)",
  rushInto: "cubic-bezier(0.55, 0, 1, 0.45)",
  rushFrom: "cubic-bezier(0, 0.55, 0.45, 1)",
  linear: "linear",
} as const;
