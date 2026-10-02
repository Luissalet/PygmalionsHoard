import { useEffect, useRef } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { useLoad } from "./ui.jsx";

// A tool call that loads on mount and whenever `deps` or the app-wide `version` change.
export function useTool(name, args, deps = []) {
  const { version } = useApp();
  return useLoad(() => api.call(name, args), [name, version, ...deps]);
}

// Runs `fn` every `ms` (`ms` null or 0 stops it). With `immediate` the first call is unconditional, hidden tab or not, so a page
// that gets its first data only through polling never shows a spinner forever; pages that load with `useLoad` leave it off.
// Ticks are skipped while the tab is hidden, and one tick runs as soon as it becomes visible again.
export function usePoll(fn, ms, immediate = false) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!ms) return undefined;
    let stopped = false;
    const run = () => { if (!stopped) ref.current(); };
    const visible = () => { if (!document.hidden) run(); };
    if (immediate) run();
    const timer = setInterval(visible, ms);
    document.addEventListener("visibilitychange", visible);
    return () => { stopped = true; clearInterval(timer); document.removeEventListener("visibilitychange", visible); };
  }, [ms, immediate]);
}

// Reads a number or text from the hash query (#/entrenar?base=a_1).
export function useQueryValue(query, key) {
  return query?.get(key) || "";
}
