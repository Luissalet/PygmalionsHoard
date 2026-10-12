// Pure helpers behind window.hoardCopilot: no DOM or React at import time, so they run under plain `node --test`.

export const CONTEXT_REQUEST = "hoard:copilot-context-request";
export const REFRESH_EVENT = "pygmalion:copilot-updated";

// A slow backend must not hold a turn back: each read gives up after this long and that part is simply left out.
export const READ_TIMEOUT_MS = 4000;

const NO_OBJECT = new Set(["", "new", "undefined", "null"]);

/** True when a route segment names no stored object ("new", empty, placeholders). */
export function isNoObjectId(id) {
  return id === undefined || id === null || NO_OBJECT.has(String(id).trim().toLowerCase());
}

/** Same grammar as the app's hash router: `#/<page>/<id>?query`. The panel is the empty page. */
export function parseRoute(hash) {
  const [path] = String(hash || "").replace(/^#\/?/, "").split("?");
  const parts = path.split("/");
  let id = parts[1] || null;
  if (id) {
    try { id = decodeURIComponent(id); } catch { /* keep the raw segment */ }
  }
  return { page: parts[0] || "panel", id: isNoObjectId(id) ? null : id };
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Keys of `current` whose value differs from `initial` (deep compare); a key missing from `current` is not reported. */
export function changedFields(initial, current) {
  const base = initial || {};
  return Object.fromEntries(Object.entries(current || {}).filter(([key, value]) => !same(value, base[key])));
}

/** `current` without the entries that are still empty (blank text, no choice, empty list/object): a create form reports what was filled in. */
export function filled(current) {
  const empty = (v) => v === undefined || v === null || v === "" || (Array.isArray(v) && !v.length) || (typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length);
  return Object.fromEntries(Object.entries(current || {}).filter(([, v]) => !empty(v)));
}

/** A form field whose saved value moved from `prev` to `next` underneath it: follow the new value unless the user has typed something else. */
export function follow(current, prev, next) {
  return same(current, prev) ? next : current;
}

export const MAX_FIELD_CHARS = 2000;

/** Deep copy of `value` where long strings (pasted JSONL, notes) are cut, so one big field cannot swamp a turn. */
export function clip(value, max = MAX_FIELD_CHARS) {
  if (typeof value === "string") return value.length > max ? `${value.slice(0, max)}… [+${value.length - max} chars]` : value;
  if (Array.isArray(value)) return value.map((v) => clip(v, max));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, clip(v, max)]));
  return value;
}

/** Snapshot a form hands to the copilot. `objectId` is omitted for create forms. */
export function formSnapshot(kind, { objectId, initial, current } = {}) {
  const fields = changedFields(initial, current);
  return { kind, ...(isNoObjectId(objectId) ? {} : { object_id: String(objectId) }), dirty: Object.keys(fields).length > 0, fields: clip(fields) };
}

/**
 * Ask every mounted form and view (they answer synchronously) and keep the most relevant form answer: the last dirty
 * one, else the last. Views that show an object the route does not name (a dataset picked in the list, a drawer
 * opened on an artifact) answer with `provideObject({ kind, object_id, ...extra })`.
 */
export function collectEditor(dispatch) {
  const answers = [];
  const objects = [];
  try {
    dispatch({
      provide: (snapshot) => { if (snapshot && typeof snapshot === "object" && snapshot.kind) answers.push(snapshot); },
      provideObject: (ref) => { if (ref && typeof ref === "object" && ref.kind && !isNoObjectId(ref.object_id)) objects.push({ ...ref, object_id: String(ref.object_id) }); },
    });
  } catch { /* a broken form must not break the turn */ }
  const dirty = answers.filter((a) => a.dirty);
  const editor = dirty.length ? dirty[dirty.length - 1] : answers[answers.length - 1];
  return { editor, objects };
}

/** Resolve with `promise`, or with `undefined` after `ms` / on rejection: one slow or failing read never blocks the others. */
export function settle(promise, ms = READ_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(undefined), ms);
    Promise.resolve(promise).then((v) => { clearTimeout(timer); resolve(v); }, () => { clearTimeout(timer); resolve(undefined); });
  });
}

const enc = (id) => encodeURIComponent(id);
const BASE_KINDS = new Set(["base", "merged", "ctx_variant"]);
const text = (v) => (v === undefined || v === null ? "" : typeof v === "object" ? String(v.text ?? v.name ?? "") : String(v));

/**
 * What each kind of object is read with. `read` gets the client API and the object reference and returns the list of
 * active objects it resolves to (empty when the id does not resolve). Every call is a real tool of the backend.
 */
export const READERS = {
  base_model: async (api, ref) => {
    // artifact_get, not base_get: the latter probes the trainer's Python environment, too heavy to run on every turn
    const a = await api.call("artifact_get", { artifact: ref.object_id });
    if (!a || !a.id || !BASE_KINDS.has(a.kind)) return [];
    return [{ kind: "base_model", object_id: String(a.id), title: text(a.name || a.id), url: `#/modelos/${enc(a.id)}` }];
  },
  dataset: async (api, ref) => {
    const r = await api.call("dataset_get", { dataset: ref.object_id, ...(ref.n ? { n: Number(ref.n) } : {}) });
    const d = r && r.dataset;
    if (!d || !d.id) return [];
    const url = `#/datasets/${enc(d.id)}`;
    const out = [{ kind: "dataset", object_id: String(d.id), title: text(d.name || d.id), url }];
    if (r.version && r.version.id) out.push({ kind: "dataset_version", object_id: String(r.version.id), title: `${text(d.name)} v${r.version.n}`, url });
    return out;
  },
  job: async (api, ref) => {
    const j = await api.call("job_get", { job: ref.object_id, log_lines: 0, curve_points: 0 });
    if (!j || !j.id) return [];
    return [{ kind: "job", object_id: String(j.id), title: `${text(j.title || j.kind || j.id)} (${text(j.state)})`, url: `#/trabajos/${enc(j.id)}` }];
  },
  artifact: async (api, ref) => {
    const a = await api.call("artifact_get", { artifact: ref.object_id });
    if (!a || !a.id) return [];
    return [{ kind: "artifact", object_id: String(a.id), title: `${text(a.name || a.id)} (${text(a.kind)})`, url: `#/linaje/${enc(a.id)}` }];
  },
};

/** Which object a route page names with its `/<id>` segment. */
export const ROUTE_KINDS = { modelos: "base_model", datasets: "dataset", trabajos: "job", linaje: "artifact" };

/**
 * Build `{ active_objects?, editor? }` for the current view. Every part is optional: a failing or slow read omits that
 * part, an id that does not resolve (or "new") means "no object", and the promise never rejects.
 *  - api: the app's client API (`api.call(tool, args)`)
 *  - ask: `(detail) => void` that dispatches the context request to the mounted forms and views
 */
export async function buildContext({ hash, api, ask, timeout = READ_TIMEOUT_MS }) {
  // Always say which objects are open (possibly none): an empty list overrides the launcher's guesses from the URL.
  const out = { active_objects: [] };
  let route = { page: "panel", id: null };
  try { route = parseRoute(hash); } catch { /* default route */ }

  let editor;
  let provided = [];
  try {
    const collected = collectEditor(ask);
    editor = collected.editor;
    provided = collected.objects;
  } catch { /* omit the editor part */ }

  // the route's object first; a view that shows the same object adds detail to it (e.g. the dataset version on screen)
  const refs = new Map();
  const add = (ref) => {
    const key = `${ref.kind}:${ref.object_id}`;
    refs.set(key, { ...(refs.get(key) || {}), ...ref });
  };
  const kind = ROUTE_KINDS[route.page];
  if (kind && route.id) add({ kind, object_id: route.id });
  for (const ref of provided) if (READERS[ref.kind]) add(ref);
  const unique = [...refs.values()];

  if (api && typeof api.call === "function" && unique.length) {
    const resolved = await Promise.all(unique.map((ref) => settle((async () => READERS[ref.kind](api, ref))(), timeout)));
    const objects = [];
    const ids = new Set();
    for (const list of resolved) {
      for (const obj of list || []) {
        const key = `${obj.kind}:${obj.object_id}`;
        if (!ids.has(key)) { ids.add(key); objects.push(obj); }
      }
    }
    out.active_objects = objects;
  }

  if (editor) out.editor = editor;
  return out;
}
