// The animated course as a self-playing video, 3Blue1Brown style: every beat (<Step>) enters
// with a manim-like animation, its narration runs as a subtitle (optionally spoken), widget
// scripts (`drive`) operate the figures, and the timeline moves on by itself at a steady pace.
// Space pauses, ← / → jump between beats, the progress bar jumps between parts.
import { renderRich } from "../../lib/rich";
import { Clock } from "../../manim/clock";
import { createSvg, fadeIn, fadeOut, indicate, morph, write } from "../../manim/animations";
import { parseDrive, runDrive } from "../../manim/drive";

interface Labels { play: string; pause: string; voiceOn: string; voiceOff: string }
interface Options { lang: "zh" | "en"; labels: Labels }

const SPEEDS = [0.75, 1, 1.25, 1.5];

export function initCourse(root: HTMLElement, { lang, labels }: Options) {
  const slides = [...root.querySelectorAll<HTMLElement>(".slide")];
  const stage = root.querySelector<HTMLElement>(".stage")!;
  const subtitle = root.querySelector<HTMLElement>(".subtitle-text")!;
  const counter = root.querySelector<HTMLElement>(".stage-counter")!;
  const playBtn = root.querySelector<HTMLButtonElement>("[data-act=play]")!;
  const voiceBtn = root.querySelector<HTMLButtonElement>("[data-act=voice]")!;
  const speedBtn = root.querySelector<HTMLButtonElement>("[data-act=speed]")!;
  const startOverlay = root.querySelector<HTMLElement>(".start-overlay")!;
  const endOverlay = root.querySelector<HTMLElement>(".end-overlay")!;
  const segments = [...root.querySelectorAll<HTMLElement>(".seg")];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const clock = new Clock();
  let slide = 0;
  let step = 0; // beats revealed on the current slide
  let playing = false;
  let voice = false;
  let speedIdx = 1;
  let run: AbortController | null = null;

  const stepsOf = (i: number) => [...slides[i].querySelectorAll<HTMLElement>(".step")];

  // ---------- layout: scale the fixed 1280×720 stage to fit ----------
  const fit = () => {
    const box = root.querySelector<HTMLElement>(".stage-fit")!;
    const s = Math.min(box.clientWidth / 1280, box.clientHeight / 720);
    stage.style.transform = `scale(${s})`;
  };
  addEventListener("resize", fit);
  fit();

  /** Shrink a tall widget (CSS zoom) so it fits the visual column of the 1280×720 canvas. */
  const fitVisual = (el: HTMLElement) => {
    const box = el.querySelector<HTMLElement>(".slide-visual");
    const inner = box?.firstElementChild as HTMLElement | null;
    if (!box || !inner) return;
    inner.style.zoom = "1";
    const avail = box.clientHeight;
    const need = inner.scrollHeight;
    if (avail > 0 && need > avail) inner.style.zoom = String(Math.max(0.6, avail / need));
  };
  addEventListener("resize", () => fitVisual(slides[slide]));

  // ---------- narration ----------
  const readingMs = (text: string) => {
    const plain = text.replace(/\$[^$]*\$/g, "x");
    return lang === "zh"
      ? Math.max(1800, plain.replace(/\s/g, "").length * 215)
      : Math.max(1800, plain.split(/\s+/).filter(Boolean).length * 330);
  };

  const showSubtitle = (text: string) => {
    subtitle.innerHTML = renderRich(text);
    subtitle.classList.toggle("empty", !text);
    if (text && !reduced) subtitle.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 350, easing: "ease-out" });
  };

  /** Show the subtitle; resolve when it has been read (or spoken). */
  function narrate(text: string, signal: AbortSignal): Promise<void> {
    showSubtitle(text);
    if (!text) return Promise.resolve();
    if (!voice || !("speechSynthesis" in window)) return clock.wait(readingMs(text), signal);
    return new Promise((resolve) => {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text.replace(/\$[^$]*\$/g, "").replace(/\*\*/g, ""));
      u.lang = lang === "zh" ? "zh-CN" : "en-US";
      u.rate = (lang === "zh" ? 1.05 : 1) * clock.speed;
      const done = () => resolve();
      u.onend = done;
      u.onerror = () => clock.wait(readingMs(text), signal).then(done);
      signal.addEventListener("abort", () => { speechSynthesis.cancel(); done(); }, { once: true });
      speechSynthesis.speak(u);
    });
  }

  // ---------- static state (used for jumps and before animating) ----------
  function applyState() {
    slides.forEach((el, i) => el.classList.toggle("active", i === slide));
    fitVisual(slides[slide]);
    const fitting = slides[slide];
    setTimeout(() => fitVisual(fitting), 400); // again once the widget has hydrated
    const steps = stepsOf(slide);
    steps.forEach((el, i) => {
      el.classList.toggle("shown", i < step);
      el.classList.remove("morph-top", "morph-out");
    });
    // In each morph group only the newest shown beat is visible.
    slides[slide].querySelectorAll<HTMLElement>(".morph-group").forEach((g) => {
      const shown = [...g.querySelectorAll<HTMLElement>(":scope > .step.shown")];
      shown.at(-1)?.classList.add("morph-top");
      shown.slice(0, -1).forEach((s) => s.classList.add("morph-out"));
    });
    root.dataset.part = slides[slide].dataset.part;
    counter.textContent = `${slide + 1} / ${slides.length}`;
    segments.forEach((seg) => {
      const [a, b] = seg.dataset.range!.split("-").map(Number);
      seg.classList.toggle("on", slide >= a && slide <= b);
      const total = b - a + 1;
      const pos = slide < a ? 0 : slide > b ? 1 : (slide - a + (step / Math.max(1, stepsOf(slide).length))) / total;
      seg.querySelector<HTMLElement>(".seg-fill")!.style.transform = `scaleX(${Math.min(1, pos)})`;
    });
    history.replaceState(null, "", `#${slide + 1}`);
  }

  const currentSay = () => {
    const shown = stepsOf(slide).slice(0, step);
    for (let i = shown.length - 1; i >= 0; i--) if (shown[i].dataset.say) return shown[i].dataset.say!;
    return slides[slide].dataset.say ?? "";
  };

  // ---------- entrances ----------
  async function enterSlide(signal: AbortSignal) {
    const s = slides[slide];
    if (reduced) return;
    const jobs: Promise<void>[] = [];
    const part = s.querySelector<HTMLElement>(".slide-part, .slide-kicker");
    const title = s.querySelector<HTMLElement>(".slide-h2, .slide-h1");
    const rule = s.querySelector<HTMLElement>(".slide-rule");
    if (part) jobs.push(fadeIn(clock, part, { dy: 8, runTime: 0.6, signal }));
    if (title) jobs.push(write(clock, title, { runTime: s.classList.contains("layout-title") ? 1.6 : 1, signal }));
    if (rule) jobs.push(clock.track(rule.animate([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 900, delay: 300, easing: "cubic-bezier(0.45,0,0.25,1)", fill: "backwards" }), signal));
    s.querySelectorAll<HTMLElement>(".slide-title-card > p").forEach((p) => jobs.push(fadeIn(clock, p, { delay: 0.9, signal })));
    const visual = s.querySelector<HTMLElement>(".slide-visual");
    if (visual) {
      jobs.push(fadeIn(clock, visual, { dy: 0, runTime: 0.5, signal }));
      jobs.push(createSvg(clock, visual, { runTime: 1.8, signal }));
    }
    s.querySelectorAll<HTMLElement>(".recap li").forEach((li, i) => jobs.push(fadeIn(clock, li, { delay: 0.3 + 0.35 * i, signal })));
    await Promise.all(jobs);
  }

  async function enterStep(el: HTMLElement, signal: AbortSignal) {
    el.classList.add("shown");
    if (reduced) return;
    const fx = el.dataset.fx ?? "write";
    if (fx === "morph") {
      const group = el.closest(".morph-group");
      const prev = group ? [...group.querySelectorAll<HTMLElement>(":scope > .step.morph-top")].find((p) => p !== el) : undefined;
      if (prev) await morph(clock, prev, el, { signal });
      else { el.classList.add("morph-top"); await write(clock, el, { signal }); }
      return;
    }
    if (fx === "fade") await fadeIn(clock, el, { signal });
    else if (fx === "pop") await fadeIn(clock, el, { dy: 0, scale: 0.8, signal });
    else {
      await write(clock, el, { signal });
      if (fx === "focus") await indicate(clock, el, { color: getComputedStyle(el).color, signal });
    }
    // Emphasised words get a manim Indicate once written.
    el.querySelectorAll<HTMLElement>("strong").forEach((b) => void indicate(clock, b, { signal }));
  }

  async function playBeat(el: HTMLElement, signal: AbortSignal) {
    const cmds = el.dataset.drive ? parseDrive(el.dataset.drive) : [];
    const visual = slides[slide].querySelector<HTMLElement>(".slide-visual") ?? slides[slide];
    const animation = (async () => {
      await enterStep(el, signal);
      if (cmds.length) await runDrive(clock, visual, stage, cmds, signal);
    })();
    await Promise.all([animation, narrate(el.dataset.say ?? "", signal)]);
    await clock.wait(450 + 1000 * +(el.dataset.hold ?? 0), signal);
  }

  // ---------- the timeline ----------
  /** Play from the current position. `fresh` = the slide has not been introduced yet. */
  async function timeline(signal: AbortSignal, fresh: boolean) {
    while (!signal.aborted) {
      if (fresh) {
        applyState();
        const intro = enterSlide(signal);
        await Promise.all([intro, narrate(slides[slide].dataset.say ?? "", signal)]);
        await clock.wait(350, signal);
        fresh = false;
        continue;
      }
      const steps = stepsOf(slide);
      if (step < steps.length) {
        step++;
        applyState();
        await playBeat(steps[step - 1], signal);
        continue;
      }
      if (slide >= slides.length - 1) { finish(); return; }
      // Clear the stage (FadeOut of everything), then the next slide.
      await Promise.all([...slides[slide].children].map((c) => (c.classList.contains("slide-bg") ? Promise.resolve() : fadeOut(clock, c, { signal }))));
      if (signal.aborted) return;
      slides[slide].getAnimations({ subtree: true }).forEach((a) => a.cancel());
      slide++;
      step = 0;
      fresh = true;
    }
  }

  function restart(fresh: boolean) {
    run?.abort();
    clock.finishAll();
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    run = new AbortController();
    if (playing) void timeline(run.signal, fresh);
  }

  function finish() {
    setPlaying(false);
    endOverlay.hidden = false;
  }

  // ---------- manual navigation ----------
  /** Show beat `step` of `slide` immediately; animate just that beat if moving forward. */
  function jumpTo(newSlide: number, newStep: number, animate: boolean) {
    run?.abort();
    clock.finishAll();
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    const changed = newSlide !== slide;
    if (changed) slides[slide].getAnimations({ subtree: true }).forEach((a) => a.cancel());
    slide = Math.max(0, Math.min(slides.length - 1, newSlide));
    const steps = stepsOf(slide);
    step = Math.max(0, Math.min(steps.length, newStep));
    endOverlay.hidden = true;
    run = new AbortController();
    const signal = run.signal;
    if (animate && step > 0) {
      // Lay out everything before the target beat, then animate the target beat itself.
      const target = step;
      step = target - 1;
      applyState();
      step = target;
      if (changed) void enterSlide(signal);
      const el = steps[step - 1];
      const cmds = el.dataset.drive ? parseDrive(el.dataset.drive) : [];
      const visual = slides[slide].querySelector<HTMLElement>(".slide-visual") ?? slides[slide];
      const beat = enterStep(el, signal).then(() => (cmds.length ? runDrive(clock, visual, stage, cmds, signal) : undefined));
      const said = playing ? narrate(el.dataset.say ?? "", signal) : (showSubtitle(el.dataset.say ?? ""), Promise.resolve());
      void Promise.all([beat, said]).then(async () => {
        if (signal.aborted) return;
        applyState();
        if (playing) { await clock.wait(450, signal); if (!signal.aborted) await timeline(signal, false); }
      });
    } else if (playing && changed && step === 0) {
      void timeline(signal, true);
    } else {
      applyState();
      if (changed && animate) void enterSlide(signal);
      showSubtitle(currentSay());
      if (playing) void clock.wait(1200, signal).then(() => { if (!signal.aborted) void timeline(signal, false); });
    }
  }

  const next = () => {
    const n = stepsOf(slide).length;
    if (step < n) jumpTo(slide, step + 1, true);
    else if (slide < slides.length - 1) jumpTo(slide + 1, 0, true);
  };
  const prev = () => {
    if (step > 0) jumpTo(slide, step - 1, false);
    else if (slide > 0) jumpTo(slide - 1, stepsOf(slide - 1).length, false);
  };

  // ---------- play / pause / voice / speed ----------
  function setPlaying(on: boolean) {
    const wasStarted = run !== null;
    playing = on;
    playBtn.textContent = on ? "❚❚" : "▶";
    playBtn.title = on ? labels.pause : labels.play;
    root.classList.toggle("playing", on);
    if (on) {
      startOverlay.hidden = true;
      endOverlay.hidden = true;
      if (clock.paused) {
        clock.setPaused(false);
        if (voice && "speechSynthesis" in window) speechSynthesis.resume();
      }
      if (!wasStarted) restart(true);
    } else {
      clock.setPaused(true);
      if ("speechSynthesis" in window) speechSynthesis.pause();
    }
  }

  playBtn.addEventListener("click", () => {
    if (!playing && endOverlay.hidden === false) { replay(); return; }
    if (!playing && run === null) { setPlaying(true); return; }
    setPlaying(!playing);
  });
  voiceBtn.addEventListener("click", () => {
    voice = !voice;
    voiceBtn.classList.toggle("on", voice);
    voiceBtn.title = voice ? labels.voiceOff : labels.voiceOn;
    if (!voice && "speechSynthesis" in window) speechSynthesis.cancel();
  });
  speedBtn.addEventListener("click", () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    clock.setSpeed(SPEEDS[speedIdx]);
    speedBtn.textContent = `${SPEEDS[speedIdx]}×`;
  });
  root.querySelector("[data-act=next]")!.addEventListener("click", next);
  root.querySelector("[data-act=prev]")!.addEventListener("click", prev);
  root.querySelector("[data-act=full]")!.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else root.requestFullscreen?.();
  });
  segments.forEach((seg) => seg.addEventListener("click", () => jumpTo(+seg.dataset.range!.split("-")[0], 0, true)));

  const start = (withVoice: boolean) => {
    if (withVoice && !voice) voiceBtn.click();
    clock.setPaused(false);
    playing = false;
    run = null;
    setPlaying(true);
  };
  root.querySelector("[data-act=start]")!.addEventListener("click", () => start(false));
  root.querySelector("[data-act=start-voice]")!.addEventListener("click", () => start(true));
  function replay() {
    endOverlay.hidden = true;
    slides[slide].getAnimations({ subtree: true }).forEach((a) => a.cancel());
    slide = 0;
    step = 0;
    applyState();
    run?.abort();
    run = null;
    clock.setPaused(false);
    setPlaying(true);
  }
  root.querySelector("[data-act=replay]")!.addEventListener("click", replay);

  stage.addEventListener("click", (e) => {
    // Clicking the canvas toggles pause, like a video; clicks on widgets and links do not.
    if ((e.target as HTMLElement).closest(".widget, a, button, input, label")) return;
    if (run === null) start(false);
    else setPlaying(!playing);
  });

  document.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).closest("input, textarea, [contenteditable]")) return;
    if (e.key === " " || e.key === "k" || e.key === "K") { e.preventDefault(); if (run === null) start(false); else setPlaying(!playing); }
    else if (["ArrowRight", "PageDown", "l", "L"].includes(e.key)) { e.preventDefault(); next(); }
    else if (["ArrowLeft", "PageUp", "j", "J"].includes(e.key)) { e.preventDefault(); prev(); }
    else if (e.key === "f" || e.key === "F") root.querySelector<HTMLElement>("[data-act=full]")!.click();
    else if (e.key === "Home") jumpTo(0, 0, true);
    else if (e.key === "End") jumpTo(slides.length - 1, 0, true);
  });

  // Touch: swipe to jump beats.
  let x0 = 0;
  stage.addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
  stage.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
  });

  // Initial state: the requested slide, fully revealed behind the start overlay.
  const fromHash = parseInt(location.hash.slice(1), 10);
  slide = Number.isFinite(fromHash) ? Math.max(0, Math.min(slides.length - 1, fromHash - 1)) : 0;
  step = 0;
  applyState();
  showSubtitle("");
  if (reduced) root.classList.add("reduced");
}
