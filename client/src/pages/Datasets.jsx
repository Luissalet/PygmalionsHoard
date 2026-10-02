import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { compact, num, seconds, splitList } from "../format.js";
import { Histogram } from "../components/charts.jsx";
import { Busy, Check, Chip, Empty, ErrorBox, Field, Icon, ICONS, KV, Modal, PageHead, Rel, Section, Seg, Tabs, useBusy, useLoad } from "../components/ui.jsx";
import { useTool } from "../components/hooks.js";

const SOURCE_TYPES = ["files", "folder", "jsonl", "csv", "family", "synthetic"];
const emptySource = (type) => ({
  type, paths: "", path: "", extensions: ".txt, .md", text: "", mode: "paste", prompt: "prompt", response: "response", textCol: "text",
  app: "", tool: "", args: "{}", items_path: "", m_prompt: "", m_response: "", m_text: "",
  task: "", fromType: "folder", fromPath: "", max_items: 100, per_chunk: 3,
});

// The source as the tools take it.
export function sourceSpec(s) {
  const mapping = (a, b, c) => Object.fromEntries([["prompt", a], ["response", b], ["text", c]].filter(([, v]) => v));
  switch (s.type) {
    case "files": return { type: "files", paths: splitList(s.paths) };
    case "folder": return { type: "folder", path: s.path.trim(), extensions: splitList(s.extensions) };
    case "jsonl": return s.mode === "paste" ? { type: "jsonl", text: s.text } : { type: "jsonl", path: s.path.trim() };
    case "csv": {
      const columns = mapping(s.prompt, s.response, s.textCol);
      return s.mode === "paste" ? { type: "csv", text: s.text, columns } : { type: "csv", path: s.path.trim(), columns };
    }
    case "family": {
      let args = {};
      try { args = JSON.parse(s.args || "{}"); } catch { args = {}; }
      return { type: "family", app: s.app.trim(), tool: s.tool.trim(), arguments: args, ...(s.items_path ? { items_path: s.items_path.trim() } : {}), mapping: mapping(s.m_prompt, s.m_response, s.m_text) };
    }
    case "synthetic": {
      const from = s.fromType === "files" ? { type: "files", paths: splitList(s.fromPath) } : { type: "folder", path: s.fromPath.trim() };
      return { type: "synthetic", task: s.task.trim(), from: [from], max_items: Number(s.max_items) || 100, per_chunk: Number(s.per_chunk) || 3 };
    }
    default: return { type: s.type };
  }
}

function sourceSummary(spec, t) {
  switch (spec.type) {
    case "files": return `${t("files_count", { n: spec.paths.length })}: ${spec.paths.slice(0, 2).join(", ")}`;
    case "folder": return spec.path;
    case "jsonl": case "csv": return spec.path || t("lines_count", { n: (spec.text || "").split("\n").filter(Boolean).length });
    case "family": return `${spec.app} / ${spec.tool}`;
    case "synthetic": return spec.task.slice(0, 80);
    default: return "";
  }
}

function SourceForm({ onAdd }) {
  const { t } = useApp();
  const [type, setType] = useState("files");
  const [s, setS] = useState(emptySource("files"));
  const [busy, run] = useBusy();
  const [preview, setPreview] = useState(null);
  const set = (k, v) => setS((x) => ({ ...x, [k]: v }));
  const pick = (next) => { setType(next); setS(emptySource(next)); setPreview(null); };
  const spec = sourceSpec({ ...s, type });
  const valid = {
    files: spec.paths?.length, folder: spec.path, jsonl: spec.text || spec.path, csv: spec.text || spec.path, family: spec.app && spec.tool && Object.keys(spec.mapping || {}).length,
    synthetic: spec.task && spec.from[0].path !== "" && (spec.from[0].paths ? spec.from[0].paths.length : true),
  }[type];
  const doPreview = () => run("p", async () => setPreview(await api.call("dataset_preview_source", { source: spec, limit: 4 })));
  const add = () => { onAdd(spec); setS(emptySource(type)); setPreview(null); };
  const textOrPath = (
    <>
      <Seg value={s.mode} onChange={(v) => set("mode", v)} options={[{ value: "paste", label: t("src_paste") }, { value: "path", label: t("src_path") }]} label={t("source")} />
      {s.mode === "paste"
        ? <Field label={t("src_text")}><textarea className="field" rows={5} value={s.text} onChange={(e) => set("text", e.target.value)} placeholder={type === "csv" ? "prompt,response\n…" : '{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'} /></Field>
        : <Field label={t("file_path")}><input className="field mono" value={s.path} onChange={(e) => set("path", e.target.value)} placeholder="/home/…/datos.jsonl" /></Field>}
    </>
  );
  return (
    <div className="space-y-3 rounded-md border p-3" style={{ borderColor: "var(--hoard-border)" }}>
      <Tabs value={type} onChange={pick} options={SOURCE_TYPES.map((v) => ({ value: v, label: t(`src_${v}`) }))} />
      <p className="help">{t(`src_${type}_help`)}</p>
      {type === "files" && <Field label={t("src_files_paths")}><textarea className="field" rows={3} value={s.paths} onChange={(e) => set("paths", e.target.value)} placeholder="/home/…/diario.md" /></Field>}
      {type === "folder" && (
        <div className="grid gap-3 sm:grid-cols-[2fr_1fr]">
          <Field label={t("folder")}><input className="field mono" value={s.path} onChange={(e) => set("path", e.target.value)} placeholder="/home/…/escritos" /></Field>
          <Field label={t("extensions")}><input className="field" value={s.extensions} onChange={(e) => set("extensions", e.target.value)} /></Field>
        </div>
      )}
      {type === "jsonl" && textOrPath}
      {type === "csv" && (
        <>
          {textOrPath}
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label={t("col_prompt")}><input className="field" value={s.prompt} onChange={(e) => set("prompt", e.target.value)} /></Field>
            <Field label={t("col_response")}><input className="field" value={s.response} onChange={(e) => set("response", e.target.value)} /></Field>
            <Field label={t("col_text")}><input className="field" value={s.textCol} onChange={(e) => set("textCol", e.target.value)} /></Field>
          </div>
        </>
      )}
      {type === "family" && (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label={t("family_app")}><input className="field" value={s.app} onChange={(e) => set("app", e.target.value)} placeholder="kafka" /></Field>
            <Field label={t("family_tool")}><input className="field" value={s.tool} onChange={(e) => set("tool", e.target.value)} placeholder="documents_list" /></Field>
            <Field label={t("items_path")} hint={t("items_path_hint")}><input className="field" value={s.items_path} onChange={(e) => set("items_path", e.target.value)} placeholder="documents" /></Field>
          </div>
          <Field label={t("family_args")}><textarea className="field" rows={2} value={s.args} onChange={(e) => set("args", e.target.value)} /></Field>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label={t("map_prompt")}><input className="field" value={s.m_prompt} onChange={(e) => set("m_prompt", e.target.value)} /></Field>
            <Field label={t("map_response")}><input className="field" value={s.m_response} onChange={(e) => set("m_response", e.target.value)} /></Field>
            <Field label={t("map_text")}><input className="field" value={s.m_text} onChange={(e) => set("m_text", e.target.value)} /></Field>
          </div>
        </>
      )}
      {type === "synthetic" && (
        <>
          <Field label={t("synth_task")} hint={t("synth_task_hint")}><textarea className="field" rows={2} value={s.task} onChange={(e) => set("task", e.target.value)} placeholder="genera 3 preguntas y respuestas sobre este fragmento" /></Field>
          <div className="grid gap-3 sm:grid-cols-4">
            <Field label={t("synth_from")}>
              <select className="field" value={s.fromType} onChange={(e) => set("fromType", e.target.value)}>
                <option value="folder">{t("src_folder")}</option>
                <option value="files">{t("src_files")}</option>
              </select>
            </Field>
            <Field label={t("file_path")} className="sm:col-span-2"><input className="field mono" value={s.fromPath} onChange={(e) => set("fromPath", e.target.value)} /></Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label={t("max_items")}><input className="field" type="number" min="1" value={s.max_items} onChange={(e) => set("max_items", e.target.value)} /></Field>
              <Field label={t("per_chunk")}><input className="field" type="number" min="1" value={s.per_chunk} onChange={(e) => set("per_chunk", e.target.value)} /></Field>
            </div>
          </div>
          <p className="help">{t("synth_pending_hint")}</p>
        </>
      )}
      <div className="flex flex-wrap gap-2">
        <Busy className="btn" busy={busy.p} disabled={!valid} onClick={doPreview}><Icon d={ICONS.search} size={14} />{t("preview")}</Busy>
        <button type="button" className="btn btn-primary" disabled={!valid} onClick={add}><Icon d={ICONS.plus} size={14} />{t("add_source")}</button>
      </div>
      {preview && <PreviewBox preview={preview} />}
    </div>
  );
}

function PreviewBox({ preview }) {
  const { t } = useApp();
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Chip className="chip-accent">{preview.count ?? preview.items?.length ?? 0} {t("items")}</Chip>
        {preview.kind && <Chip>{preview.kind}</Chip>}
        {(preview.notes || []).map((n, i) => <Chip key={i} className="chip-amber chip-wrap">{t.msg(n)}</Chip>)}
      </div>
      {preview.estimate && <div className="help">{t("synthetic_estimate", { calls: preview.estimate.calls, max: preview.estimate.max_items, time: seconds(preview.estimate.estimated_seconds) })} {t.msg(preview.estimate.note)}</div>}
      <div className="logbox">{(preview.items || []).map((it) => JSON.stringify({ ...it, _meta: undefined }, null, 0)).join("\n")}</div>
    </div>
  );
}

const defaultOps = () => ({ dedupe_exact: true, dedupe_near: false, near: 0.85, length: false, min_chars: 20, max_chars: 8000, language: false, keep_es: true, keep_en: true, pii: "" });

export function opsList(o) {
  const list = [];
  if (o.dedupe_exact) list.push({ op: "dedupe_exact" });
  if (o.dedupe_near) list.push({ op: "dedupe_near", threshold: Number(o.near) || 0.85 });
  if (o.length) list.push({ op: "length", min_chars: Number(o.min_chars) || 0, max_chars: Number(o.max_chars) || 100000 });
  if (o.language) list.push({ op: "language", keep: [o.keep_es && "es", o.keep_en && "en"].filter(Boolean) });
  if (o.pii) list.push({ op: "pii", mode: o.pii });
  return list;
}

export function OpsForm({ value, onChange }) {
  const { t } = useApp();
  const set = (k, v) => onChange({ ...value, [k]: v });
  return (
    <div className="space-y-2">
      <Check checked={value.dedupe_exact} onChange={(v) => set("dedupe_exact", v)}>{t("op_dedupe_exact")}</Check><br />
      <Check checked={value.dedupe_near} onChange={(v) => set("dedupe_near", v)}>{t("op_dedupe_near")}</Check>
      {value.dedupe_near && <input className="field ml-2" style={{ width: 80 }} type="number" min="0.5" max="1" step="0.01" value={value.near} onChange={(e) => set("near", e.target.value)} aria-label={t("threshold")} />}
      <br />
      <Check checked={value.length} onChange={(v) => set("length", v)}>{t("op_length")}</Check>
      {value.length && (
        <span className="ml-2 inline-flex items-center gap-2">
          <input className="field" style={{ width: 90 }} type="number" min="0" value={value.min_chars} onChange={(e) => set("min_chars", e.target.value)} aria-label={t("min_chars")} />–
          <input className="field" style={{ width: 100 }} type="number" min="1" value={value.max_chars} onChange={(e) => set("max_chars", e.target.value)} aria-label={t("max_chars")} />
        </span>
      )}
      <br />
      <Check checked={value.language} onChange={(v) => set("language", v)}>{t("op_language")}</Check>
      {value.language && <span className="ml-2 inline-flex gap-3"><Check checked={value.keep_es} onChange={(v) => set("keep_es", v)}>es</Check><Check checked={value.keep_en} onChange={(v) => set("keep_en", v)}>en</Check></span>}
      <br />
      <label className="check">{t("op_pii")}
        <select className="field" style={{ width: "auto" }} value={value.pii} onChange={(e) => set("pii", e.target.value)}>
          <option value="">{t("pii_off")}</option>
          <option value="mask">{t("pii_mask")}</option>
          <option value="drop">{t("pii_drop")}</option>
        </select>
      </label>
    </div>
  );
}

function Builder({ existing, onClose, onDone }) {
  const { t, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [sources, setSources] = useState([]);
  const [ops, setOps] = useState(defaultOps());
  const [split, setSplit] = useState({ eval_pct: 5, min_eval: 20, seed: 42 });
  const [error, setError] = useState(null);
  const build = () => run("b", async () => {
    setError(null);
    try {
      const args = { sources, operations: opsList(ops), split: { eval_pct: Number(split.eval_pct), min_eval: Number(split.min_eval), seed: Number(split.seed) }, wait_s: 20 };
      if (existing) args.dataset = existing.id; else args.name = name.trim();
      if (description) args.description = description;
      const r = await api.call("dataset_create", args);
      notify(r.done ? t("dataset_built") : t("job_started", { id: r.job?.id || "" }));
      changed();
      onDone(r);
    } catch (e) {
      setError(e);
    }
  });
  return (
    <Modal title={existing ? t("new_version_of", { name: existing.name }) : t("new_dataset")} onClose={onClose} wide>
      <ErrorBox error={error} />
      {!existing && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} placeholder="Mis escritos" /></Field>
          <Field label={t("description")}><input className="field" value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
        </div>
      )}
      <div className="space-y-2">
        <h3>{t("sources")} <span className="chip">{sources.length}</span></h3>
        {sources.map((s, i) => (
          <div key={i} className="panel panel-tight flex items-center gap-2">
            <Chip className="chip-accent">{t(`src_${s.type}`)}</Chip>
            <span className="trunc min-w-0 flex-1">{sourceSummary(s, t)}</span>
            <button type="button" className="btn btn-sm" onClick={() => setSources(sources.filter((_, j) => j !== i))} aria-label={t("remove")}><Icon d={ICONS.x} size={12} /></button>
          </div>
        ))}
        <SourceForm onAdd={(spec) => setSources([...sources, spec])} />
      </div>
      <div className="space-y-2">
        <h3>{t("operations")}</h3>
        <OpsForm value={ops} onChange={setOps} />
      </div>
      <div className="space-y-2">
        <h3>{t("split")}</h3>
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label={t("eval_pct")}><input className="field" type="number" min="0" max="50" step="0.5" value={split.eval_pct} onChange={(e) => setSplit({ ...split, eval_pct: e.target.value })} /></Field>
          <Field label={t("min_eval")}><input className="field" type="number" min="0" value={split.min_eval} onChange={(e) => setSplit({ ...split, min_eval: e.target.value })} /></Field>
          <Field label={t("seed")}><input className="field" type="number" value={split.seed} onChange={(e) => setSplit({ ...split, seed: e.target.value })} /></Field>
        </div>
      </div>
      <div className="flex justify-end gap-2">
        <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
        <Busy className="btn btn-primary" busy={busy.b} disabled={!sources.length || (!existing && !name.trim())} onClick={build}>{t("build_dataset")}</Busy>
      </div>
    </Modal>
  );
}

function Stats({ stats }) {
  const { t, lang } = useApp();
  if (!stats) return null;
  const roles = stats.roles || {};
  const total = Object.values(roles).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="panel space-y-3">
      <div className="kpis">
        <div className="kpi"><b className="num">{num(stats.records, 0, lang)}</b><span>{t("records")}</span></div>
        <div className="kpi"><b className="num">{num(stats.usable, 0, lang)}</b><span>{t("usable")}</span></div>
        <div className="kpi"><b className="num">{compact(stats.tokens, lang)}</b><span>{t("tokens_est")}</span></div>
        <div className="kpi"><b className="num">{compact(stats.chars, lang)}</b><span>{t("chars")}</span></div>
        <div className="kpi"><b className="num">{num(stats.median_chars, 0, lang)}</b><span>{t("median_len")}</span></div>
        <div className="kpi"><b className="num">{num(stats.pending, 0, lang)}</b><span>{t("pending")}</span></div>
        <div className="kpi"><b className="num">{num(stats.rejected, 0, lang)}</b><span>{t("rejected")}</span></div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div><h3 className="mb-1">{t("length_hist")}</h3><Histogram buckets={stats.histogram} ariaLabel={t("length_hist")} /></div>
        <div className="space-y-2">
          <h3>{t("role_balance")}</h3>
          {Object.entries(roles).length ? Object.entries(roles).map(([role, n]) => (
            <div key={role} className="flex items-center gap-2"><span style={{ minWidth: 80 }}>{role}</span><div className="bar flex-1"><i style={{ width: `${(n / total) * 100}%` }} /></div><span className="num help">{num(n, 0, lang)}</span></div>
          )) : <div className="help">—</div>}
          <h3>{t("languages")}</h3>
          <div className="flex flex-wrap gap-1.5">{Object.entries(stats.languages || {}).map(([k, n]) => <Chip key={k}>{k}: {num(n, 0, lang)}</Chip>)}</div>
          <h3>PII</h3>
          <div className="help">{stats.pii?.records_with_pii ? t("pii_found", { n: stats.pii.records_with_pii, kinds: Object.keys(stats.pii.by_kind || {}).join(", ") }) : t("pii_none")}</div>
        </div>
      </div>
      {stats.report?.length > 0 && (
        <details>
          <summary>{t("build_report")}</summary>
          <table className="mt-2"><thead><tr><th>{t("operation")}</th><th className="r">{t("before")}</th><th className="r">{t("after")}</th><th className="r">{t("changed")}</th></tr></thead>
            <tbody>{stats.report.map((r, i) => <tr key={i}><td>{r.op}</td><td className="r num">{r.before}</td><td className="r num">{r.after}</td><td className="r num">{r.changed}</td></tr>)}</tbody></table>
        </details>
      )}
      {stats.notes?.length > 0 && stats.notes.map((n, i) => <div key={i} className="banner banner-info">{t.msg(n)}</div>)}
    </div>
  );
}

function RecordEditor({ record, onClose, onSave, busy }) {
  const { t } = useApp();
  const [text, setText] = useState(JSON.stringify(record.record, null, 2));
  const [err, setErr] = useState("");
  const save = () => {
    try { onSave(JSON.parse(text)); } catch { setErr(t("bad_json")); }
  };
  return (
    <Modal title={t("edit_record", { n: record.index })} onClose={onClose} wide>
      <textarea className="field" rows={14} value={text} onChange={(e) => { setText(e.target.value); setErr(""); }} aria-label={t("record")} />
      {err && <div className="banner banner-danger">{err}</div>}
      <div className="flex justify-end gap-2">
        <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
        <Busy className="btn btn-primary" busy={busy} onClick={save}>{t("save")}</Busy>
      </div>
    </Modal>
  );
}

const PAGE = 20;
function Records({ dataset, n, pending, onChanged }) {
  const { t, notify, changed, version } = useApp();
  const [filters, setFilters] = useState({ status: "", text: "", split: "", lang: "" });
  const [offset, setOffset] = useState(0);
  const [editing, setEditing] = useState(null);
  const [busy, run] = useBusy();
  const set = (k, v) => { setFilters((f) => ({ ...f, [k]: v })); setOffset(0); };
  const { data, error, reload } = useLoad(() => api.call("dataset_records", { dataset, n: n || undefined, offset, limit: PAGE, ...Object.fromEntries(Object.entries(filters).filter(([, v]) => v)) }), [dataset, n, offset, JSON.stringify(filters), version]);
  const review = (body, message) => run("r", async () => {
    const r = await api.call("dataset_review", { dataset, n: n || undefined, ...body });
    notify(message || t("review_saved", { n: r.version?.n ?? "" }));
    setEditing(null);
    changed();
    onChanged?.(r);
  });
  const rows = data?.records || [];
  return (
    <div className="space-y-2">
      <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto_auto_auto] sm:items-end">
        <Field label={t("search")}><input className="field" value={filters.text} onChange={(e) => set("text", e.target.value)} /></Field>
        <Field label={t("state")}>
          <select className="field" value={filters.status} onChange={(e) => set("status", e.target.value)}>
            <option value="">{t("all")}</option>
            {["ok", "pending", "rejected"].map((s) => <option key={s} value={s}>{t(`rec_${s}`)}</option>)}
          </select>
        </Field>
        <Field label={t("split_col")}>
          <select className="field" value={filters.split} onChange={(e) => set("split", e.target.value)}>
            <option value="">{t("all")}</option>
            <option value="train">train</option>
            <option value="eval">eval</option>
          </select>
        </Field>
        <Field label={t("lang_col")}>
          <select className="field" value={filters.lang} onChange={(e) => set("lang", e.target.value)}>
            <option value="">{t("all")}</option>
            <option value="es">es</option>
            <option value="en">en</option>
          </select>
        </Field>
        {pending > 0 && <Busy className="btn btn-primary" busy={busy.r} onClick={() => review({ accept_all_pending: true })}><Icon d={ICONS.check} size={14} />{t("accept_all_pending", { n: pending })}</Busy>}
      </div>
      <ErrorBox error={error} />
      <div className="scroll-x">
        <table>
          <thead><tr><th className="r">#</th><th>{t("record")}</th><th>{t("state")}</th><th>{t("split_col")}</th><th className="r">{t("chars")}</th><th /></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.index}>
                <td className="r num">{r.index}</td>
                <td style={{ overflowWrap: "anywhere", maxWidth: 520 }}><div className="clamp2">{r.preview}</div><div className="help">{r.source}{r.synthetic ? ` · ${t("synthetic")}` : ""}</div></td>
                <td><Chip className={r.status === "ok" ? "chip-ok" : r.status === "pending" ? "chip-amber" : "chip-danger"}>{t(`rec_${r.status}`)}</Chip></td>
                <td>{r.split || "—"}</td>
                <td className="r num">{r.chars}</td>
                <td className="r" style={{ whiteSpace: "nowrap" }}>
                  {r.status !== "ok" && <button type="button" className="btn btn-sm" disabled={busy.r} onClick={() => review({ accept: [r.index] }, t("record_accepted"))} aria-label={t("accept")}><Icon d={ICONS.check} size={12} /></button>}{" "}
                  {r.status !== "rejected" && <button type="button" className="btn btn-sm" disabled={busy.r} onClick={() => review({ reject: [r.index] }, t("record_rejected"))} aria-label={t("reject")}><Icon d={ICONS.x} size={12} /></button>}{" "}
                  <button type="button" className="btn btn-sm" onClick={() => setEditing(r)} aria-label={t("edit")}><Icon d={ICONS.pencil} size={12} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data && !rows.length && <div className="help">{t("no_records")}</div>}
      {data && (
        <div className="flex items-center gap-2">
          <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}><Icon d={ICONS.back} size={12} />{t("prev")}</button>
          <span className="help num">{data.total ? `${offset + 1}–${Math.min(offset + PAGE, data.total)} / ${data.total}` : "0"}</span>
          <button type="button" className="btn btn-sm" disabled={offset + PAGE >= (data.total || 0)} onClick={() => setOffset(offset + PAGE)}>{t("next")}<Icon d={ICONS.chevron} size={12} /></button>
        </div>
      )}
      {editing && <RecordEditor record={editing} busy={busy.r} onClose={() => setEditing(null)} onSave={(rec) => review({ edit: { [editing.index]: rec } }, t("record_edited"))} />}
    </div>
  );
}

function DatasetDetail({ id, onGone }) {
  const { t, notify, confirm, changed, version } = useApp();
  const [n, setN] = useState("");
  const [busy, run] = useBusy();
  const [ops, setOps] = useState(defaultOps());
  const [showOps, setShowOps] = useState(false);
  const [newVersion, setNewVersion] = useState(false);
  const { data, error, reload } = useLoad(() => api.call("dataset_get", { dataset: id, n: n || undefined }), [id, n, version]);
  useEffect(() => { setN(""); }, [id]);
  if (!data) return <div className="help">{error ? <ErrorBox error={error} /> : t("loading")}</div>;
  const ds = data.dataset;
  const v = data.version;
  const remove = async () => {
    if (!(await confirm({ title: t("delete_dataset"), message: t("delete_dataset_msg", { name: ds.name }) }))) return;
    run("d", async () => { await api.call("dataset_delete", { dataset: ds.id, confirm: true }); notify(t("deleted")); changed(); onGone(); });
  };
  const apply = () => run("a", async () => {
    const r = await api.call("dataset_apply", { dataset: ds.id, n: v.n, operations: opsList(ops) });
    notify(t("ops_applied", { n: r.version?.n ?? "" }));
    setShowOps(false);
    changed();
    setN("");
  });
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 style={{ fontSize: 18 }}>{ds.name}</h2>
        <Chip>{ds.kind}</Chip>
        <Rel ts={ds.updated_ts} />
        <div className="ml-auto flex flex-wrap gap-2">
          <select className="field" style={{ width: "auto" }} value={n || v.n} onChange={(e) => setN(e.target.value)} aria-label={t("version")}>
            {ds.version_list.map((x) => <option key={x.id} value={x.n}>v{x.n} · {x.usable} {t("records_short")}</option>)}
          </select>
          <button type="button" className="btn btn-sm" onClick={() => setNewVersion(true)}><Icon d={ICONS.plus} size={13} />{t("new_version")}</button>
          <button type="button" className="btn btn-sm" onClick={() => setShowOps(!showOps)}>{t("apply_ops")}</button>
          <a className="btn btn-sm btn-primary" href={`#/entrenar?dataset=${ds.id}`}><Icon d={ICONS.bolt} size={13} />{t("train_with")}</a>
          <Busy className="btn btn-sm btn-danger" busy={busy.d} onClick={remove} aria-label={t("delete")}><Icon d={ICONS.trash} size={13} /></Busy>
        </div>
      </div>
      {ds.description && <p className="help">{ds.description}</p>}
      {showOps && (
        <div className="panel space-y-3">
          <OpsForm value={ops} onChange={setOps} />
          <div className="flex gap-2"><Busy className="btn btn-primary" busy={busy.a} onClick={apply}>{t("apply_ops")}</Busy><button type="button" className="btn" onClick={() => setShowOps(false)}>{t("cancel")}</button></div>
          <p className="help">{t("ops_new_version_hint")}</p>
        </div>
      )}
      <Stats stats={v.stats} />
      <div className="grid gap-3 md:grid-cols-2">
        <div className="panel space-y-1">
          <h3>{t("recipe")}</h3>
          <KV items={[
            [t("version"), `v${v.n}`], [t("hash"), <span className="mono">{String(v.sha256).slice(0, 16)}…</span>],
            [t("split"), data.splits ? `${data.splits.eval_pct} % · min ${data.splits.min_eval} · seed ${data.splits.seed}` : ""],
            [t("train_eval"), `${v.train} / ${v.eval}`],
          ]} />
          <details><summary>JSON</summary><pre className="logbox mt-2">{JSON.stringify(data.recipe, null, 2)}</pre></details>
        </div>
      </div>
      <Section id="records" title={t("records")} count={v.records}>
        <Records dataset={ds.id} n={v.n} pending={v.pending} />
      </Section>
      {newVersion && <Builder existing={ds} onClose={() => setNewVersion(false)} onDone={() => { setNewVersion(false); setN(""); }} />}
    </div>
  );
}

export default function Datasets({ param }) {
  const { t, lang } = useApp();
  const { data, error, loading } = useTool("datasets_list", {}, []);
  const [selected, setSelected] = useState(param || null);
  const [building, setBuilding] = useState(false);
  const list = data?.datasets || [];
  useEffect(() => { if (!selected && list.length) setSelected(list[0].id); }, [list.length]);
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_datasets")} subtitle={t("datasets_sub")} actions={<button type="button" className="btn btn-primary" onClick={() => setBuilding(true)}><Icon d={ICONS.plus} size={14} />{t("new_dataset")}</button>} />
      <ErrorBox error={error} />
      {loading && !data ? <div className="help">{t("loading")}</div> : list.length ? (
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <div className="space-y-2">
            {list.map((d) => (
              <button key={d.id} type="button" className="panel pick space-y-1" aria-pressed={selected === d.id} onClick={() => setSelected(d.id)}>
                <div className="font-semibold" style={{ overflowWrap: "anywhere" }}>{d.name}</div>
                <div className="help flex flex-wrap gap-x-3">
                  <span>{d.kind}</span>
                  {d.latest && <span className="num">{num(d.latest.usable, 0, lang)} {t("records_short")}</span>}
                  {d.latest && <span className="num">{compact(d.latest.tokens, lang)} tok</span>}
                  <span>{d.versions} {t("versions_short")}</span>
                </div>
                {d.latest?.pending > 0 && <Chip className="chip-amber">{t("pending_n", { n: d.latest.pending })}</Chip>}
              </button>
            ))}
          </div>
          {selected && <DatasetDetail id={selected} onGone={() => setSelected(null)} />}
        </div>
      ) : <Empty>{t("no_datasets")}</Empty>}
      {building && <Builder onClose={() => setBuilding(false)} onDone={(r) => { setBuilding(false); const id = r.result?.dataset?.id; if (id) setSelected(id); }} />}
    </div>
  );
}
