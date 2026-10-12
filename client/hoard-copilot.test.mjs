import test from "node:test";
import assert from "node:assert/strict";
import {
  READERS, buildContext, changedFields, clip, collectEditor, filled, follow, formSnapshot, isNoObjectId, parseRoute, settle,
} from "./src/copilot-core.js";

const dataset = { id: "ds_1", name: "Mis escritos", kind: "chat", version_list: [{ id: "dv_1", n: 1 }, { id: "dv_2", n: 2 }] };
const versions = { 1: { id: "dv_1", n: 1 }, 2: { id: "dv_2", n: 2 } };
const artifacts = {
  a_base: { id: "a_base", name: "Qwen3-4B", kind: "base" },
  a_adapter: { id: "a_adapter", name: "estilo-v1", kind: "adapter", notes: "n" },
};
const tools = {
  artifact_get: ({ artifact }) => artifacts[artifact] || Promise.reject(Object.assign(new Error("unknown artifact"), { status: 404 })),
  dataset_get: ({ dataset: id, n }) => {
    if (id !== "ds_1") throw Object.assign(new Error("unknown dataset"), { status: 404 });
    return { dataset, version: versions[n || 2] };
  },
  job_get: ({ job }) => {
    if (job !== "job_1") throw Object.assign(new Error("unknown job"), { status: 404 });
    return { id: "job_1", kind: "train", title: "Train estilo-v1", state: "running" };
  },
};
const apiOf = (calls = []) => ({ call: async (name, args) => { calls.push([name, args]); return tools[name](args || {}); } });
const noForms = () => {};
const summary = (ctx) => (ctx.active_objects || []).map((o) => [o.kind, o.object_id, o.title]);

test("parseRoute follows the hash router; the panel is the empty page and new/blank ids are no object", () => {
  assert.deepEqual(parseRoute("#/trabajos/job_1"), { page: "trabajos", id: "job_1" });
  assert.deepEqual(parseRoute("#/datasets/mis%20datos?x=1"), { page: "datasets", id: "mis datos" });
  assert.deepEqual(parseRoute("#/entrenar?base=a_1"), { page: "entrenar", id: null });
  assert.deepEqual(parseRoute("#/linaje/new"), { page: "linaje", id: null });
  assert.deepEqual(parseRoute("#/linaje/"), { page: "linaje", id: null });
  assert.deepEqual(parseRoute("#/"), { page: "panel", id: null });
  assert.deepEqual(parseRoute(undefined), { page: "panel", id: null });
  assert.equal(isNoObjectId("NEW"), true);
  assert.equal(isNoObjectId("ds_1"), false);
});

test("changedFields reports only differing keys, deeply, including cleared values", () => {
  assert.deepEqual(changedFields({ a: 1, b: [1, 2], c: "x" }, { a: 1, b: [1, 3], c: "" }), { b: [1, 3], c: "" });
  assert.deepEqual(changedFields(undefined, { a: 1 }), { a: 1 });
  assert.deepEqual(changedFields({ a: 1 }, { a: 1 }), {});
});

test("filled keeps what a create form has filled in and drops empties", () => {
  assert.deepEqual(filled({ base: "a_base", dataset: "", rank: 16, after: undefined, targets: [], values: {}, force: false }), { base: "a_base", rank: 16, force: false });
});

test("formSnapshot: edit forms carry object_id, create forms do not, long text is cut", () => {
  const edit = formSnapshot("artifact-form", { objectId: "a_1", initial: { name: "A", notes: "" }, current: { name: "B", notes: "" } });
  assert.deepEqual(edit, { kind: "artifact-form", object_id: "a_1", dirty: true, fields: { name: "B" } });
  const draft = formSnapshot("training-draft", { initial: {}, current: { base: "a_base" } });
  assert.deepEqual(draft, { kind: "training-draft", dirty: true, fields: { base: "a_base" } });
  assert.equal("object_id" in formSnapshot("training-draft", { objectId: "new", initial: {}, current: {} }), false);
  assert.equal(formSnapshot("x", { objectId: "p", initial: { a: 1 }, current: { a: 1 } }).dirty, false);
  const big = formSnapshot("dataset-draft", { initial: {}, current: { sources: [{ type: "jsonl", text: "x".repeat(5000) }] } });
  assert.match(big.fields.sources[0].text, /^x{2000}… \[\+3000 chars\]$/);
  assert.equal(clip("short"), "short");
});

test("follow keeps unsaved typing when the saved value changes underneath", () => {
  assert.equal(follow("old", "old", "agent edit"), "agent edit");     // untouched field adopts the new saved value
  assert.equal(follow("my text", "old", "agent edit"), "my text");   // typed field is kept
  assert.equal(follow("same", "same", "same"), "same");
});

test("collectEditor prefers the last dirty answer and gathers the objects views report", () => {
  const { editor, objects } = collectEditor(({ provide, provideObject }) => {
    provide({ kind: "artifact-form", dirty: false, fields: {} });
    provide({ kind: "training-draft", dirty: true, fields: { base: "a_base" } });
    provide({ kind: "other", dirty: false, fields: {} });
    provideObject({ kind: "dataset", object_id: "ds_1", n: 2 });
    provideObject({ kind: "dataset", object_id: "new" });
    provideObject({ kind: "artifact" });
    provide(null);
  });
  assert.equal(editor.kind, "training-draft");
  assert.deepEqual(objects, [{ kind: "dataset", object_id: "ds_1", n: 2 }]);
  assert.equal(collectEditor(() => { throw new Error("boom"); }).editor, undefined);
});

test("each route resolves its object through the real tools", async () => {
  const calls = [];
  const api = apiOf(calls);
  assert.deepEqual(summary(await buildContext({ hash: "#/trabajos/job_1", api, ask: noForms })), [["job", "job_1", "Train estilo-v1 (running)"]]);
  assert.deepEqual(calls.pop(), ["job_get", { job: "job_1", log_lines: 0, curve_points: 0 }]);
  assert.deepEqual(summary(await buildContext({ hash: "#/linaje/a_adapter", api, ask: noForms })), [["artifact", "a_adapter", "estilo-v1 (adapter)"]]);
  assert.deepEqual(calls.pop(), ["artifact_get", { artifact: "a_adapter" }]);
  assert.deepEqual(summary(await buildContext({ hash: "#/modelos/a_base", api, ask: noForms })), [["base_model", "a_base", "Qwen3-4B"]]);
  assert.deepEqual(summary(await buildContext({ hash: "#/datasets/ds_1", api, ask: noForms })), [["dataset", "ds_1", "Mis escritos"], ["dataset_version", "dv_2", "Mis escritos v2"]]);
  assert.deepEqual(calls.pop(), ["dataset_get", { dataset: "ds_1" }]);
});

test("a view's own object (picked in a list) is read, and its version wins over the route's duplicate", async () => {
  const calls = [];
  const ctx = await buildContext({ hash: "#/datasets", api: apiOf(calls), ask: ({ provideObject }) => provideObject({ kind: "dataset", object_id: "ds_1", n: 1 }) });
  assert.deepEqual(summary(ctx), [["dataset", "ds_1", "Mis escritos"], ["dataset_version", "dv_1", "Mis escritos v1"]]);
  assert.deepEqual(calls, [["dataset_get", { dataset: "ds_1", n: 1 }]]);
  calls.length = 0;
  const both = await buildContext({ hash: "#/datasets/ds_1", api: apiOf(calls), ask: ({ provideObject }) => { provideObject({ kind: "dataset", object_id: "ds_1", n: 2 }); provideObject({ kind: "mystery", object_id: "x" }); } });
  assert.equal(calls.length, 1);
  assert.equal(both.active_objects.length, 2);
});

test("a base route whose id is not a base model resolves to no object", async () => {
  assert.deepEqual(await buildContext({ hash: "#/modelos/a_adapter", api: apiOf(), ask: noForms }), { active_objects: [] });
});

test("new, unresolvable ids and failing reads give no object and never throw", async () => {
  const calls = [];
  const api = apiOf(calls);
  assert.deepEqual(await buildContext({ hash: "#/datasets/new", api, ask: noForms }), { active_objects: [] });
  assert.deepEqual(await buildContext({ hash: "#/linaje/", api, ask: noForms }), { active_objects: [] });
  assert.deepEqual(calls, []);
  for (const hash of ["#/datasets/ghost", "#/trabajos/ghost", "#/linaje/ghost", "#/modelos/ghost"]) assert.deepEqual(await buildContext({ hash, api, ask: noForms }), { active_objects: [] }, hash);
  assert.deepEqual(await buildContext({ hash: "#/linaje/x", api: { call: async () => null }, ask: noForms }), { active_objects: [] });
  assert.deepEqual(await buildContext({ hash: "#/linaje/x", api: { call: () => { throw new Error("sync"); } }, ask: noForms }), { active_objects: [] });
});

test("one failing object does not drop the others, and a slow read is abandoned", async () => {
  const api = { call: async (name, args) => {
    if (name === "job_get") return new Promise(() => {});          // never answers
    return tools[name](args);
  } };
  const ctx = await buildContext({ hash: "#/trabajos/job_1", api, timeout: 20, ask: ({ provideObject }) => { provideObject({ kind: "artifact", object_id: "a_base" }); provideObject({ kind: "dataset", object_id: "ghost" }); } });
  assert.deepEqual(summary(ctx), [["artifact", "a_base", "Qwen3-4B (base)"]]);
  assert.equal(await settle(new Promise(() => {}), 10), undefined);
  assert.equal(await settle(Promise.reject(new Error("x")), 10), undefined);
  assert.equal(await settle(Promise.resolve(3), 10), 3);
});

test("create forms are reported on any route, including with no object", async () => {
  const ctx = await buildContext({ hash: "#/entrenar?base=a_base", api: apiOf(), ask: ({ provide }) => {
    provide(formSnapshot("training-draft", { initial: {}, current: filled({ base: "a_base", dataset: "", rank: 8 }) }));
  } });
  assert.deepEqual(ctx, { active_objects: [], editor: { kind: "training-draft", dirty: true, fields: { base: "a_base", rank: 8 } } });
});

test("an edit form on a detail route is reported next to its object", async () => {
  const ctx = await buildContext({ hash: "#/linaje/a_adapter", api: apiOf(), ask: ({ provide }) => {
    provide(formSnapshot("artifact-form", { objectId: "a_adapter", initial: { name: "estilo-v1", notes: "n" }, current: { name: "estilo-v1", notes: "mine" } }));
  } });
  assert.deepEqual(ctx.editor, { kind: "artifact-form", object_id: "a_adapter", dirty: true, fields: { notes: "mine" } });
  assert.equal(ctx.active_objects.length, 1);
});

test("a failing form or a missing API leaves context() resolved", async () => {
  assert.deepEqual(await buildContext({ hash: "#/linaje/a_base", api: { call: () => Promise.reject(new Error("down")) }, ask: () => { throw new Error("form"); } }), { active_objects: [] });
  assert.deepEqual(await buildContext({ hash: undefined, api: {}, ask: noForms }), { active_objects: [] });
  assert.deepEqual(await buildContext({ hash: "#/linaje/a_base", api: undefined, ask: noForms }), { active_objects: [] });
});

test("every tool the readers call exists in the backend catalogue", async () => {
  const { readFileSync } = await import("node:fs");
  const catalogue = readFileSync(new URL("../pygmalion_hoard/agent_tools.py", import.meta.url), "utf8");
  const names = new Set([...catalogue.matchAll(/^\s*Tool\("([a-z_]+)"/gm)].map((m) => m[1]));
  const used = [...readFileSync(new URL("./src/copilot-core.js", import.meta.url), "utf8").matchAll(/api\.call\("([a-z_]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...new Set(used)].sort(), ["artifact_get", "dataset_get", "job_get"]);
  for (const name of used) assert.ok(names.has(name), name);
  assert.deepEqual(Object.keys(READERS).sort(), ["artifact", "base_model", "dataset", "job"]);
});

test("window.hoardCopilot: context() never rejects and onTurnEnd fires the refresh event", async () => {
  const target = new EventTarget();
  const hits = [];
  globalThis.window = Object.assign(target, { location: { hash: "#/datasets/new", origin: "http://x.test" } });
  globalThis.fetch = async () => { throw new Error("offline"); };
  await import("./src/hoard-copilot.js");
  const hc = globalThis.window.hoardCopilot;
  assert.equal(typeof hc.context, "function");
  target.addEventListener("hoard:copilot-context-request", (e) => e.detail.provide({ kind: "dataset-draft", dirty: true, fields: { name: "N" } }));
  assert.deepEqual(await hc.context(), { active_objects: [], view: { editor: { kind: "dataset-draft", dirty: true, fields: { name: "N" } } } });
  globalThis.window.location.hash = "#/datasets/ds_9";      // backend unreachable: api rejects, context still resolves
  assert.deepEqual(await hc.context(), { active_objects: [], view: { editor: { kind: "dataset-draft", dirty: true, fields: { name: "N" } } } });
  target.addEventListener("pygmalion:copilot-updated", () => hits.push(1));
  hc.onTurnEnd();
  assert.equal(hits.length, 1);
});


test("a job whose title is a translatable object reads as its text", async () => {
  const api = { call: async (tool) => (tool === "job_get" ? { id: "j_1", kind: "evaluate", state: "done", title: { key: "title_evaluate", params: {}, text: "evaluate m-Q4" } } : null) };
  const ctx = await buildContext({ hash: "#/trabajos/j_1", api, ask: () => {} });
  assert.equal(ctx.active_objects[0].title, "evaluate m-Q4 (done)");
});
