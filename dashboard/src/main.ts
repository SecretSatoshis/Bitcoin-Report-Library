import type { ChartPayload } from "./data";
const toggle = document.querySelector<HTMLButtonElement>("#nav-toggle")!;
const nav = document.querySelector<HTMLElement>("#site-links")!;
function closeMenu() {
  nav.removeAttribute("data-open");
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", "Open menu");
}
toggle.addEventListener("click", () => {
  const open = !nav.hasAttribute("data-open");
  nav.toggleAttribute("data-open", open);
  toggle.setAttribute("aria-expanded", String(open));
  toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
});
nav.addEventListener("click", closeMenu);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (nav.hasAttribute("data-open")) {
      closeMenu();
      toggle.focus();
    }
    document.querySelectorAll(".help").forEach((el) => {
      el.removeAttribute("data-open");
      el.setAttribute("data-dismissed", "");
      el.querySelector("button")?.setAttribute("aria-expanded", "false");
    });
  }
});
document
  .querySelectorAll<HTMLButtonElement>(".help-button")
  .forEach((button, i) => {
    const container = button.parentElement!,
      description = container.querySelector<HTMLElement>(".help-text")!;
    description.id = `metric-help-${i}`;
    button.setAttribute("aria-describedby", description.id);
    button.addEventListener("focus", () =>
      container.removeAttribute("data-dismissed"),
    );
    container.addEventListener("mouseenter", () =>
      container.removeAttribute("data-dismissed"),
    );
    container.addEventListener("mouseleave", () =>
      container.removeAttribute("data-dismissed"),
    );
    button.addEventListener("click", () => {
      const open = !container.hasAttribute("data-open");
      container.toggleAttribute("data-open", open);
      container.toggleAttribute("data-dismissed", !open);
      button.setAttribute("aria-expanded", String(open));
    });
  });
const links = document.querySelectorAll<HTMLAnchorElement>(".section-nav a");
const sections = new IntersectionObserver(
  (entries) => {
    for (const entry of entries)
      if (entry.isIntersecting)
        links.forEach((link) => {
          if (link.hash === "#" + entry.target.id)
            link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
  },
  { rootMargin: "-140px 0px -55% 0px" },
);
document
  .querySelectorAll(".dashboard-section")
  .forEach((section) => sections.observe(section));
const payloads = new Map<string, Promise<ChartPayload>>();
function loadPayload(frame: HTMLIFrameElement): Promise<ChartPayload> {
  const id = frame.dataset.chartFrame!;
  if (!payloads.has(id))
    payloads.set(
      id,
      fetch(frame.dataset.chartSrc!, { cache: "no-cache" }).then((response) => {
        if (!response.ok) throw new Error(`chart data returned ${response.status}`);
        return response.json() as Promise<ChartPayload>;
      }),
    );
  return payloads.get(id)!;
}
const frames = [
  ...document.querySelectorAll<HTMLIFrameElement>("[data-chart-frame]"),
];
const timers = new Map<string, ReturnType<typeof setTimeout>>();
function fail(id: string, message: string) {
  clearTimeout(timers.get(id));
  const error = document.querySelector<HTMLElement>(
    `[data-chart-error="${id}"]`,
  )!;
  error.hidden = false;
  error.textContent = `Chart unavailable: ${message}`;
  if (id === "dashboard-price-outlook")
    error.setAttribute("data-price-outlook-error", "");
  document
    .querySelector(`[data-chart-frame="${id}"]`)
    ?.setAttribute("data-chart-ready", "failed");
}
async function initialize(frame: HTMLIFrameElement) {
  const id = frame.dataset.chartFrame!;
  let payload: ChartPayload;
  try {
    payload = await loadPayload(frame);
    if (payload.id !== id) throw new Error("chart data belongs to another chart");
  } catch (error) {
    fail(id, (error as Error).message);
    return;
  }
  frame.contentWindow?.postMessage(
    { type: "ss-chart-init", payload },
    location.origin,
  );
  if (!timers.has(id))
    timers.set(
      id,
      setTimeout(() => fail(id, "Renderer did not become ready"), 30000),
    );
}
window.addEventListener("message", (event) => {
  if (event.origin !== location.origin) return;
  const frame = frames.find((f) => f.contentWindow === event.source),
    message = event.data;
  if (!frame || message?.id !== frame.dataset.chartFrame) return;
  if (
    message.type === "ss-chart-size" &&
    Number.isFinite(message.height) &&
    message.height >= 300 &&
    message.height <= 4000
  )
    frame.style.height = `${message.height}px`;
  if (message.type === "ss-chart-ready") {
    if (
      message.reportDate !==
      document.querySelector<HTMLElement>("[data-dashboard-date]")!.dataset
        .dashboardDate
    )
      fail(message.id, "Report date mismatch");
    else {
      clearTimeout(timers.get(message.id));
      frame.dataset.chartReady = "true";
    }
  }
  if (message.type === "ss-chart-error") fail(message.id, message.message);
});
frames.forEach((frame) => {
  frame.addEventListener("load", () => initialize(frame));
  if (frame.contentDocument?.readyState === "complete") initialize(frame);
});
