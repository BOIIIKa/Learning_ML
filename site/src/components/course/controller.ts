// Drives the animated course: slide/step navigation, narration subtitles, autoplay and
// optional text-to-speech. Works on the static HTML produced by Slide.astro / Step.astro.
import { renderRich } from "../../lib/rich";

interface Options {
  lang: "zh" | "en";
  labels: { play: string; pause: string; voiceOn: string; voiceOff: string };
}

export function initCourse(root: HTMLElement, { lang, labels }: Options) {
  const slides = [...root.querySelectorAll<HTMLElement>(".slide")];
  const stage = root.querySelector<HTMLElement>(".stage")!;
  const subtitle = root.querySelector<HTMLElement>(".subtitle-text")!;
  const counter = root.querySelector<HTMLElement>(".stage-counter")!;
  const playBtn = root.querySelector<HTMLButtonElement>("[data-act=play]")!;
  const voiceBtn = root.querySelector<HTMLButtonElement>("[data-act=voice]")!;
  const segments = [...root.querySelectorAll<HTMLElement>(".seg")];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let slide = 0;
  let step = 0; // number of revealed steps on the current slide
  let playing = false;
  let voice = false;
  let timer = 0;
  let speakToken = 0;

  const stepsOf = (i: number) => [...slides[i].querySelectorAll<HTMLElement>(".step")];

  // ---- layout: scale the fixed 1280×720 stage to fit the viewport ----
  const fit = () => {
    const box = root.querySelector<HTMLElement>(".stage-fit")!;
    const s = Math.min((box.clientWidth - 32) / 1280, (box.clientHeight - 24) / 720);
    stage.style.transform = `scale(${s})`;
  };
  addEventListener("resize", fit);
  fit();

  // ---- narration ----
  /** Narration for the current state: the latest revealed step's `say`, else the slide's. */
  const narration = (): { text: string; own: boolean } => {
    const revealed = stepsOf(slide).slice(0, step);
    for (let i = revealed.length - 1; i >= 0; i--) {
      if (revealed[i].dataset.say) return { text: revealed[i].dataset.say!, own: i === revealed.length - 1 };
    }
    return { text: slides[slide].dataset.say ?? "", own: step === 0 };
  };

  const readingMs = (text: string) =>
    lang === "zh" ? Math.max(2600, text.length * 190) : Math.max(2600, text.split(/\s+/).length * 330);

  const speak = ({ text, own }: { text: string; own: boolean }, done: () => void) => {
    const token = ++speakToken;
    if (!own || !text) {
      // Nothing new to say: give the animation a moment, then move on.
      timer = window.setTimeout(done, 1600);
      return;
    }
    if (!voice || !("speechSynthesis" in window)) {
      timer = window.setTimeout(done, readingMs(text));
      return;
    }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/\$[^$]*\$/g, ""));
    u.lang = lang === "zh" ? "zh-CN" : "en-US";
    u.rate = lang === "zh" ? 1.05 : 1;
    u.onend = () => token === speakToken && (timer = window.setTimeout(done, 500));
    u.onerror = () => token === speakToken && (timer = window.setTimeout(done, readingMs(text)));
    speechSynthesis.speak(u);
  };

  const showSubtitle = (text: string) => {
    subtitle.classList.remove("in");
    void subtitle.offsetWidth; // restart the CSS animation
    subtitle.innerHTML = renderRich(text);
    subtitle.classList.toggle("empty", !text);
    subtitle.classList.add("in");
  };

  // ---- rendering ----
  const render = (direction: 1 | -1 | 0) => {
    slides.forEach((el, i) => {
      el.classList.toggle("active", i === slide);
      el.classList.toggle("from-left", i === slide && direction < 0);
    });
    stepsOf(slide).forEach((el, i) => {
      el.classList.toggle("shown", i < step);
      el.classList.toggle("current", i === step - 1);
    });
    const part = slides[slide].dataset.part;
    segments.forEach((seg) => {
      const range = seg.dataset.range!.split("-").map(Number);
      seg.classList.toggle("on", slide >= range[0] && slide <= range[1]);
      seg.classList.toggle("done", slide > range[1]);
      const fill = seg.querySelector<HTMLElement>(".seg-fill")!;
      const pos = slide < range[0] ? 0 : slide > range[1] ? 1 : (slide - range[0] + 1) / (range[1] - range[0] + 1);
      fill.style.transform = `scaleX(${pos})`;
    });
    root.dataset.part = part;
    counter.textContent = `${slide + 1} / ${slides.length}`;
    history.replaceState(null, "", `#${slide + 1}`);
    const said = narration();
    showSubtitle(said.text);
    clearTimeout(timer);
    if (playing) speak(said, next);
    else if (voice && said.own && said.text) speak(said, () => {});
  };

  const go = (i: number, direction: 1 | -1 | 0) => {
    slide = Math.max(0, Math.min(slides.length - 1, i));
    step = 0;
    render(direction);
  };

  function next() {
    if (step < stepsOf(slide).length) {
      step++;
      render(0);
    } else if (slide < slides.length - 1) {
      go(slide + 1, 1);
    } else {
      setPlaying(false);
    }
  }

  function prev() {
    if (step > 0) {
      step--;
      render(0);
    } else if (slide > 0) {
      slide--;
      step = stepsOf(slide).length;
      render(-1);
    }
  }

  function setPlaying(on: boolean) {
    playing = on;
    clearTimeout(timer);
    if (!on && "speechSynthesis" in window) speechSynthesis.cancel();
    playBtn.textContent = on ? "❚❚" : "▶";
    playBtn.title = on ? labels.pause : labels.play;
    root.classList.toggle("playing", on);
    if (on) speak(narration(), next);
  }

  // ---- input ----
  root.querySelector("[data-act=next]")!.addEventListener("click", () => { setPlaying(false); next(); });
  root.querySelector("[data-act=prev]")!.addEventListener("click", () => { setPlaying(false); prev(); });
  playBtn.addEventListener("click", () => setPlaying(!playing));
  voiceBtn.addEventListener("click", () => {
    voice = !voice;
    voiceBtn.classList.toggle("on", voice);
    voiceBtn.title = voice ? labels.voiceOff : labels.voiceOn;
    if (!voice && "speechSynthesis" in window) speechSynthesis.cancel();
  });
  root.querySelector("[data-act=full]")!.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else root.requestFullscreen?.();
  });
  segments.forEach((seg) => seg.addEventListener("click", () => { setPlaying(false); go(+seg.dataset.range!.split("-")[0], 1); }));
  stage.addEventListener("click", (e) => {
    // Clicking empty slide space advances; clicks inside widgets do not.
    if ((e.target as HTMLElement).closest(".widget, a, button, input, .slide-visual")) return;
    setPlaying(false);
    next();
  });

  document.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).closest("input, textarea, [contenteditable]")) return;
    if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); setPlaying(false); next(); }
    else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); setPlaying(false); prev(); }
    else if (e.key === "p" || e.key === "P") setPlaying(!playing);
    else if (e.key === "f" || e.key === "F") root.querySelector<HTMLElement>("[data-act=full]")!.click();
    else if (e.key === "Home") go(0, -1);
    else if (e.key === "End") go(slides.length - 1, 1);
  });

  // Touch swipe
  let x0 = 0;
  stage.addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
  stage.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 50) { setPlaying(false); dx < 0 ? next() : prev(); }
  });

  const fromHash = parseInt(location.hash.slice(1), 10);
  go(Number.isFinite(fromHash) ? fromHash - 1 : 0, 0);
  if (reduced) root.classList.add("reduced");
}
