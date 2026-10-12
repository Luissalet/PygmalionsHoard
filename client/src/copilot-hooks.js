import { useEffect, useRef } from "react";
import { CONTEXT_REQUEST, REFRESH_EVENT, formSnapshot } from "./copilot-core.js";

/** Run `fn` (an existing reload function) when an agent turn ends. Nothing is remounted. */
export function useCopilotRefresh(fn) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    const run = () => { try { ref.current?.(); } catch { /* a reload must not break the others */ } };
    window.addEventListener(REFRESH_EVENT, run);
    return () => window.removeEventListener(REFRESH_EVENT, run);
  }, []);
}

/**
 * Let the copilot see a form's unsaved changes. `initial` is what the form started from (or its empty defaults for a
 * create form: pass no `objectId` then) and `current` its live values. A falsy `kind` (the form is not ready) stays silent.
 */
export function useCopilotForm(kind, { objectId, initial, current }) {
  const state = useRef({ kind, objectId, initial, current });
  state.current = { kind, objectId, initial, current };
  useEffect(() => {
    const answer = (event) => {
      const s = state.current;
      if (s.kind) event.detail?.provide?.(formSnapshot(s.kind, s));
    };
    window.addEventListener(CONTEXT_REQUEST, answer);
    return () => window.removeEventListener(CONTEXT_REQUEST, answer);
  }, []);
}

/**
 * Report a pending piece of the surrounding form (a sub-form whose content is not in the parent's state yet) under
 * `name`: the copilot sees it inside the form snapshot's fields. A null/undefined value stays silent.
 */
export function useCopilotPart(name, value) {
  const state = useRef({ name, value });
  state.current = { name, value };
  useEffect(() => {
    const answer = (event) => {
      const s = state.current;
      if (s.value !== undefined && s.value !== null) event.detail?.providePart?.(s.name, s.value);
    };
    window.addEventListener(CONTEXT_REQUEST, answer);
    return () => window.removeEventListener(CONTEXT_REQUEST, answer);
  }, []);
}

/**
 * Tell the copilot which object a view shows when the route does not say so (a dataset picked in the list, a drawer
 * opened on an artifact). `extra` rides along, e.g. `{ n: 3 }` for the dataset version on screen. A falsy id stays silent.
 */
export function useCopilotObject(kind, objectId, extra) {
  const state = useRef({ kind, objectId, extra });
  state.current = { kind, objectId, extra };
  useEffect(() => {
    const answer = (event) => {
      const s = state.current;
      if (s.objectId) event.detail?.provideObject?.({ ...(s.extra || {}), kind: s.kind, object_id: s.objectId });
    };
    window.addEventListener(CONTEXT_REQUEST, answer);
    return () => window.removeEventListener(CONTEXT_REQUEST, answer);
  }, []);
}
