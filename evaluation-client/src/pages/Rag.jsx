import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Drawer, Empty, ErrorBox, Field, Icon, ICONS, Rel, Section, Spinner, Tabs, useBusy, useLoad } from "../components/ui.jsx";
import { usePoll } from "../components/hooks.js";
import { Progress, StateChip } from "../components/parts.jsx";
import { num, pct } from "../format.js";

const ACTIVE = new Set(["queued", "running"]);
const PIECES = ["retriever", "ranking", "generator", "unknown"];
const PIECE_CHIP = { retriever: "chip-danger", ranking: "chip-amber", generator: "chip-amber", unknown: "" };
const CASE_CHIP = { accepted: "chip-ok", proposed: "chip-amber", rejected: "chip-danger" };
const CHANGE_CHIP = { improved: "chip-ok", regressed: "chip-danger", same_ok: "", same_failed: "chip-amber" };
const DEFAULT_FORM = {
  kind: "builtin", retriever: "bm25", chunk_chars: 600, chunk_overlap: 80, depth: 10, reranker: "none", top_k: 3, generator: "model", model: "",
  url: "", source: "documents", answer: "faustus", method: "POST", body: '{"question": "{question}", "k": "{k}"}', passages_path: "", answer_path: "",
  judge: "", judge_mode: "auto", label: "",
};
const share = (v, lang) => (v === null || v === undefined ? "—" : pct(v, 0, lang));
const signed = (v, lang) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${num(v, 2, lang)}`);

function useServers() {
  const { version } = useApp();
  const load = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  return useMemo(() => (load.data?.models || []).filter((m) => m.kind === "server"), [load.data]);
}

function ModelSelect({ value, onChange, servers, empty }) {
  return (
    <select className="field" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{empty}</option>
      {servers.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
    </select>
  );
}

// ------------------------------------------------------------------ a new set
function NewSet({ onCreated }) {
  const { t } = useApp();
  const [name, setName] = useState("");
  const [docs, setDocs] = useState("");
  const [paths, setPaths] = useState("");
  const [busy, run] = useBusy();
  const create = () => run("create", async () => {
    const documents = docs.split(/^\s*---\s*$/m).map((x) => x.trim()).filter(Boolean).map((text, i) => {
      const first = text.split("\n")[0];
      const titled = /^#\s+\S/.test(first);
      return { name: titled ? first.replace(/^#\s+/, "").trim() : `doc-${i + 1}`, text };
    });
    const out = await api.call("rag_set_save", { name, documents, paths: paths.split("\n").map((p) => p.trim()).filter(Boolean) });
    setName(""); setDocs(""); setPaths("");
    onCreated(out.set.id);
  });
  return (
    <div className="panel space-y-2">
      <p className="help">{t("rag_new_help")}</p>
      <Field label={t("rag_set_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field>
      <Field label={t("rag_docs")} hint={t("rag_docs_hint")}><textarea className="field" rows={7} value={docs} onChange={(e) => setDocs(e.target.value)} spellCheck={false} /></Field>
      <Field label={t("rag_paths")} hint={t("rag_paths_hint")}><textarea className="field mono" rows={2} value={paths} onChange={(e) => setPaths(e.target.value)} spellCheck={false} /></Field>
      <Busy className="btn btn-primary" busy={busy.create} disabled={!name.trim() || (!docs.trim() && !paths.trim())} onClick={create}><Icon d={ICONS.plus} size={14} />{t("rag_create")}</Busy>
    </div>
  );
}

// ------------------------------------------------------------------ cases
function CaseEditor({ setId, initial, onDone, onCancel }) {
  const { t } = useApp();
  const [f, setF] = useState(() => ({ question: initial?.question || "", expected: initial?.expected || "", passage: (initial?.passages || []).map((p) => p.text || p.doc || p.id).join("\n---\n"),
    answerable: initial ? initial.answerable : true }));
  const [busy, run] = useBusy();
  const save = () => run("save", async () => {
    const passages = f.answerable ? f.passage.split(/^\s*---\s*$/m).map((x) => x.trim()).filter(Boolean).map((text) => ({ text })) : [];
    await api.call("rag_case_save", { set: setId, id: initial?.id || "", question: f.question, expected: f.answerable ? f.expected : "", passages, answerable: f.answerable, status: "accepted" });
    onDone();
  });
  return (
    <div className="panel space-y-2">
      <Field label={t("rag_question")}><input className="field" value={f.question} onChange={(e) => setF({ ...f, question: e.target.value })} /></Field>
      <label className="flex items-center gap-2"><input type="checkbox" checked={f.answerable} onChange={(e) => setF({ ...f, answerable: e.target.checked })} />{t("rag_answerable")}</label>
      {f.answerable && <Field label={t("rag_expected")}><textarea className="field" rows={2} value={f.expected} onChange={(e) => setF({ ...f, expected: e.target.value })} /></Field>}
      {f.answerable && <Field label={t("rag_passages")} hint={t("rag_passages_hint")}><textarea className="field" rows={3} value={f.passage} onChange={(e) => setF({ ...f, passage: e.target.value })} /></Field>}
      <div className="flex gap-2">
        <Busy className="btn btn-primary" busy={busy.save} disabled={!f.question.trim()} onClick={save}><Icon d={ICONS.check} size={14} />{t("rag_save_accept")}</Busy>
        {onCancel && <button type="button" className="btn" onClick={onCancel}>{t("cancel")}</button>}
      </div>
    </div>
  );
}

function Cases({ detail, servers, reload }) {
  const { t, notify, confirm } = useApp();
  const set = detail.set;
  const [gen, setGen] = useState({ model: "", n: 10, unanswerable: 0 });
  const [editing, setEditing] = useState(null);
  const [adding, setAdding] = useState(false);
  const [busy, run] = useBusy();
  const generate = () => run("generate", async () => {
    const out = await api.call("rag_cases_generate", { set: set.id, model: gen.model, n: Number(gen.n), unanswerable: Number(gen.unanswerable) });
    notify(t("rag_generated", { n: out.proposed }));
    reload();
  });
  const review = (status, ids) => run(`review-${ids ? ids[0] : "all"}`, async () => {
    await api.call("rag_cases_review", ids ? { set: set.id, status, ids } : { set: set.id, status, all_proposed: true });
    reload();
  });
  const remove = (c) => run(`del-${c.id}`, async () => {
    if (!(await confirm({ title: t("rag_case_delete"), message: c.question, confirmLabel: t("delete"), danger: true }))) return;
    await api.call("rag_case_remove", { id: c.id, confirm: true });
    reload();
  });
  return (
    <div className="space-y-4">
      <Section title={t("rag_generate")}>
        <div className="panel space-y-2">
          <p className="help">{t("rag_generate_help")}</p>
          <div className="flex flex-wrap items-end gap-3">
            <Field label={t("rag_model")}><ModelSelect value={gen.model} onChange={(model) => setGen({ ...gen, model })} servers={servers} empty={t("rag_pick_model")} /></Field>
            <Field label={t("rag_n_cases")}><input className="field" type="number" min="1" max="100" value={gen.n} onChange={(e) => setGen({ ...gen, n: e.target.value })} style={{ width: 90 }} /></Field>
            <Field label={t("rag_n_unanswerable")}><input className="field" type="number" min="0" max="30" value={gen.unanswerable} onChange={(e) => setGen({ ...gen, unanswerable: e.target.value })} style={{ width: 90 }} /></Field>
            <Busy className="btn btn-primary" busy={busy.generate} disabled={!gen.model || !set.documents.length} onClick={generate}><Icon d={ICONS.plus} size={14} />{t("rag_generate_go")}</Busy>
          </div>
          {!servers.length && <p className="help">{t("rag_no_servers")}</p>}
        </div>
      </Section>
      <Section title={t("rag_cases")} count={detail.cases.length} actions={<>
        {set.cases.proposed > 0 && <Busy className="btn btn-primary" busy={busy["review-all"]} onClick={() => review("accepted")}><Icon d={ICONS.check} size={14} />{t("rag_accept_all", { n: set.cases.proposed })}</Busy>}
        <button type="button" className="btn" onClick={() => setAdding(!adding)}><Icon d={ICONS.plus} size={14} />{t("rag_add_case")}</button>
      </>}>
        {adding && <CaseEditor setId={set.id} onDone={() => { setAdding(false); reload(); }} onCancel={() => setAdding(false)} />}
        {!detail.cases.length && <Empty>{t("rag_no_cases")}</Empty>}
        <div className="space-y-2">
          {detail.cases.map((c) => (editing === c.id
            ? <CaseEditor key={c.id} setId={set.id} initial={c} onDone={() => { setEditing(null); reload(); }} onCancel={() => setEditing(null)} />
            : (
              <div key={c.id} className="panel space-y-1" data-case={c.id} data-status={c.status}>
                <div className="flex flex-wrap items-start gap-2">
                  <b className="mr-auto" style={{ flex: "1 1 320px" }}>{c.question}</b>
                  <Chip className={CASE_CHIP[c.status]}>{t(`rag_state_${c.status}`)}</Chip>
                  {!c.answerable && <Chip>{t("rag_unanswerable")}</Chip>}
                  {c.status !== "accepted" && <Busy className="btn btn-sm" busy={busy[`review-${c.id}`]} onClick={() => review("accepted", [c.id])}>{t("rag_accept")}</Busy>}
                  {c.status !== "rejected" && <Busy className="btn btn-sm" busy={busy[`review-${c.id}`]} onClick={() => review("rejected", [c.id])}>{t("rag_reject")}</Busy>}
                  <button type="button" className="btn btn-sm" onClick={() => setEditing(c.id)} aria-label={t("rag_edit")}><Icon d={ICONS.pencil} size={13} /></button>
                  <Busy className="btn btn-sm btn-danger" busy={busy[`del-${c.id}`]} onClick={() => remove(c)} aria-label={t("delete")}><Icon d={ICONS.trash} size={13} /></Busy>
                </div>
                {c.answerable && <div><span className="help">{t("rag_expected")}: </span>{c.expected || "—"}</div>}
                {c.passages.map((p, i) => <div key={i} className="help" style={{ borderLeft: "2px solid var(--line)", paddingLeft: 8 }}>{p.doc ? <b>{p.doc}: </b> : null}{p.text || p.id || ""}</div>)}
              </div>
            )))}
        </div>
      </Section>
    </div>
  );
}

// ------------------------------------------------------------------ the pipeline form
function pipelineOf(f) {
  const n = (v) => Number(v);
  if (f.kind === "builtin") {
    const needsModel = f.generator !== "extractive" || f.reranker === "model";
    return { kind: "builtin", retriever: f.retriever, reranker: f.reranker, generator: f.generator, chunk_chars: n(f.chunk_chars), chunk_overlap: n(f.chunk_overlap), depth: n(f.depth),
      top_k: n(f.top_k), model: needsModel ? f.model : "" };
  }
  if (f.kind === "faustus") return { kind: "faustus", url: f.url.trim(), source: f.source, depth: n(f.depth), answer: f.answer, model: f.answer === "model" ? f.model : "" };
  return { kind: "http", url: f.url.trim(), method: f.method, body: JSON.parse(f.body || "{}"), passages_path: f.passages_path.trim(), answer_path: f.answer_path.trim(), depth: n(f.depth),
    top_k: n(f.top_k), model: f.model };
}

function Passages({ rows, empty }) {
  if (!rows || !rows.length) return <div className="help">{empty}</div>;
  return (
    <ol className="space-y-1" style={{ margin: 0, paddingLeft: 20 }}>
      {rows.map((p, i) => (
        <li key={i} style={{ overflowWrap: "anywhere" }}>
          {p.hit ? <Chip className="chip-ok">★</Chip> : null} {p.doc ? <b>{p.doc} </b> : null}
          <span className="help">{p.score !== undefined ? `(${p.score}) ` : ""}</span>{(p.text || "").slice(0, 420)}{(p.text || "").length > 420 ? "…" : ""}
        </li>
      ))}
    </ol>
  );
}

function Evaluate({ detail, servers, onStarted }) {
  const { t, notify } = useApp();
  const set = detail.set;
  const [f, setF] = useState(DEFAULT_FORM);
  const [probe, setProbe] = useState(null);
  const [busy, run] = useBusy();
  const up = (key) => (e) => setF({ ...f, [key]: e.target.value });
  const build = () => { try { return pipelineOf(f); } catch { notify(t("rag_bad_json"), "error"); return null; } };
  const accepted = detail.cases.filter((c) => c.status === "accepted");
  const start = () => run("start", async () => {
    const pipeline = build();
    if (!pipeline) return;
    const out = await api.call("rag_run_start", { set: set.id, pipeline, judge: f.judge, judge_mode: f.judge_mode, label: f.label });
    onStarted(out.run.id);
  });
  const tryOne = () => run("probe", async () => {
    const pipeline = build();
    if (!pipeline) return;
    setProbe(await api.call("rag_probe", { set: set.id, pipeline, question: (accepted[0] || detail.cases[0]).question }));
  });
  const sel = (key, options, prefix) => (
    <select className="field" value={f[key]} onChange={up(key)}>{options.map((o) => <option key={o} value={o}>{t(`${prefix}${o}`)}</option>)}</select>
  );
  const numField = (key, label, min, max) => <Field label={label}><input className="field" type="number" min={min} max={max} value={f[key]} onChange={up(key)} style={{ width: 100 }} /></Field>;
  const needsModel = f.kind === "builtin" ? (f.generator !== "extractive" || f.reranker === "model") : f.kind === "faustus" ? f.answer === "model" : false;
  return (
    <div className="space-y-4">
      <Section title={t("rag_pipeline")}>
        <div className="panel space-y-3">
          <Field label={t("rag_kind")} hint={t(`rag_kind_help_${f.kind}`)}>{sel("kind", ["builtin", "faustus", "http"], "rag_kind_")}</Field>
          {f.kind === "builtin" && (
            <div className="flex flex-wrap items-end gap-3">
              <Field label={t("rag_retriever")}>{sel("retriever", ["bm25", "random", "none"], "rag_retriever_")}</Field>
              {numField("chunk_chars", t("rag_chunk_chars"), 80, 8000)}
              {numField("chunk_overlap", t("rag_chunk_overlap"), 0, 4000)}
              {numField("depth", t("rag_depth"), 1, 50)}
              <Field label={t("rag_reranker")}>{sel("reranker", ["none", "overlap", "model", "reverse", "random"], "rag_reranker_")}</Field>
              {numField("top_k", t("rag_top_k"), 1, 50)}
              <Field label={t("rag_generator")}>{sel("generator", ["model", "extractive", "no_context"], "rag_generator_")}</Field>
            </div>
          )}
          {f.kind === "faustus" && (
            <div className="flex flex-wrap items-end gap-3">
              <Field label={t("rag_url")} hint={t("rag_faustus_url_hint")}><input className="field" value={f.url} onChange={up("url")} placeholder="http://127.0.0.1:7000" style={{ minWidth: 240 }} /></Field>
              <Field label={t("rag_source")}>{sel("source", ["documents", "brain"], "rag_source_")}</Field>
              {numField("depth", t("rag_depth"), 1, 50)}
              <Field label={t("rag_answer_by")}>{sel("answer", ["faustus", "model"], "rag_answer_")}</Field>
            </div>
          )}
          {f.kind === "http" && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-end gap-3">
                <Field label={t("rag_url")}><input className="field" value={f.url} onChange={up("url")} placeholder="http://127.0.0.1:8000/query" style={{ minWidth: 280 }} /></Field>
                <Field label={t("rag_method")}><select className="field" value={f.method} onChange={up("method")}><option>POST</option><option>GET</option></select></Field>
                {numField("depth", t("rag_depth"), 1, 50)}
                {numField("top_k", t("rag_top_k"), 1, 50)}
              </div>
              <Field label={t("rag_body")} hint={t("rag_body_hint")}><textarea className="field mono" rows={3} value={f.body} onChange={up("body")} spellCheck={false} /></Field>
              <div className="flex flex-wrap items-end gap-3">
                <Field label={t("rag_passages_path")} hint={t("rag_path_hint")}><input className="field mono" value={f.passages_path} onChange={up("passages_path")} placeholder="results" /></Field>
                <Field label={t("rag_answer_path")}><input className="field mono" value={f.answer_path} onChange={up("answer_path")} placeholder="answer" /></Field>
              </div>
            </div>
          )}
          {(needsModel || f.kind === "http") && (
            <Field label={f.kind === "http" ? t("rag_model_optional") : t("rag_model")}><ModelSelect value={f.model} onChange={(model) => setF({ ...f, model })} servers={servers} empty={t("rag_pick_model")} /></Field>
          )}
          <div className="flex flex-wrap items-end gap-3">
            <Field label={t("rag_judge")} hint={t("rag_judge_hint")}><ModelSelect value={f.judge} onChange={(judge) => setF({ ...f, judge })} servers={servers} empty={t("rag_judge_default")} /></Field>
            <Field label={t("rag_judge_mode")}>{sel("judge_mode", ["auto", "logit", "model", "lexical"], "rag_judge_mode_")}</Field>
            <Field label={t("rag_label")}><input className="field" value={f.label} onChange={up("label")} /></Field>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Busy className="btn btn-primary" busy={busy.start} disabled={!accepted.length || (needsModel && !f.model)} onClick={start}><Icon d={ICONS.run} size={14} />{t("rag_run_go", { n: accepted.length })}</Busy>
            <Busy className="btn" busy={busy.probe} disabled={!detail.cases.length || (needsModel && !f.model)} onClick={tryOne}><Icon d={ICONS.eye} size={14} />{t("rag_probe")}</Busy>
            {!accepted.length && <span className="help">{t("rag_need_accepted")}</span>}
          </div>
        </div>
      </Section>
      {probe && (
        <Section title={t("rag_probe_result")}>
          <div className="panel space-y-2" data-probe="1">
            <div><span className="help">{t("rag_question")}: </span><b>{probe.question}</b></div>
            <div><span className="help">{t("rag_answer")}: </span>{probe.answer || "—"}</div>
            <div className="help">{t("rag_context")} ({(probe.context || []).length})</div>
            <Passages rows={probe.context} empty={t("rag_nothing")} />
            <div className="help">{t("rag_retrieved")} ({(probe.retrieved || []).length})</div>
            <Passages rows={probe.retrieved} empty={probe.traced ? t("rag_nothing") : t("rag_untraced")} />
          </div>
        </Section>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ a run
function CaseTrace({ run, caseId, onClose }) {
  const { t, lang } = useApp();
  const { data, error } = useLoad(() => api.call("rag_run_result", { run, case: caseId }), [run, caseId]);
  const c = data?.case;
  return (
    <Drawer title={c ? c.question : t("rag_case")} onClose={onClose}>
      <ErrorBox error={error} />
      {!c && !error && <Spinner />}
      {c && (
        <div className="space-y-3" data-trace={caseId}>
          <div className="flex flex-wrap gap-2">
            <Chip className={c.ok ? "chip-ok" : "chip-danger"}>{c.ok ? t("rag_ok") : t("rag_failed")}</Chip>
            {c.piece ? <Chip className={PIECE_CHIP[c.piece]}>{t(`rag_piece_${c.piece}`)}</Chip> : null}
            {c.cause ? <Chip>{t(`rag_cause_${c.cause}`)}</Chip> : null}
            {c.generator_note ? <Chip>{t(`rag_note_${c.generator_note}`)}</Chip> : null}
          </div>
          {c.error ? <div className="banner banner-danger">{c.error}</div> : null}
          <div><div className="label">{t("rag_answer")}</div><div className="pre">{c.answer || "—"}</div></div>
          {c.answerable ? <div><div className="label">{t("rag_expected")}</div><div className="pre">{c.expected || "—"}</div></div> : <div className="help">{t("rag_unanswerable_help")}</div>}
          <div className="help">
            {t("rag_faithfulness")}: <b>{share(c.faithfulness, lang)}</b> · {t("rag_coverage")}: <b>{share(c.coverage, lang)}</b> · {t("rag_abstained")}: <b>{c.abstained ? t("rag_yes") : t("rag_no")}</b>
            {c.judge_mode ? <> · {t("rag_judged_by")}: {t(`rag_judge_mode_${c.judge_mode}`)}</> : null}
          </div>
          {(c.references || []).length > 0 && <div><div className="label">{t("rag_references")}</div><Passages rows={c.references} empty="" /></div>}
          <div><div className="label">{t("rag_context")} ({(c.context || []).length})</div><Passages rows={c.context} empty={t("rag_nothing")} /></div>
          {c.ranked && <div><div className="label">{t("rag_ranked")} ({c.ranked.length})</div><Passages rows={c.ranked} empty={t("rag_nothing")} /></div>}
          <div><div className="label">{t("rag_retrieved")} ({(c.retrieved || []).length})</div><Passages rows={c.retrieved} empty={t("rag_nothing")} /></div>
          <p className="help">{t("rag_hit_help")}</p>
        </div>
      )}
    </Drawer>
  );
}

function Attribution({ a }) {
  const { t, lang } = useApp();
  if (!a) return null;
  return (
    <div className="panel space-y-2" data-attribution={a.main || "none"}>
      <b>{t("rag_attribution")}</b>
      {a.failed === 0 ? <div className="help">{t("rag_no_failures")}</div> : (
        <>
          <div data-sentence="1">{t("rag_attr_sentence", { pct: share(a.share[a.main], lang), piece: t(`rag_piece_of_${a.main}`), failed: a.failed })}</div>
          {PIECES.filter((p) => a.by_piece[p]).map((p) => (
            <div key={p} className="flex items-center gap-2">
              <span style={{ width: 150 }}>{t(`rag_piece_${p}`)}</span>
              <div style={{ flex: 1 }}><Progress done={a.by_piece[p]} total={a.failed} tone="bad" /></div>
              <span className="num" style={{ width: 90, textAlign: "right" }}>{a.by_piece[p]} · {share(a.share[p], lang)}</span>
            </div>
          ))}
          <div className="flex flex-wrap gap-1">{Object.entries(a.by_cause).map(([cause, n]) => <Chip key={cause}>{t(`rag_cause_${cause}`)}: {n}</Chip>)}</div>
          {Object.keys(a.generator_notes || {}).length > 0 && (
            <div className="help">{t("rag_without_passage")}: {Object.entries(a.generator_notes).map(([note, n]) => `${t(`rag_note_${note}`)} ${n}`).join(" · ")}</div>
          )}
        </>
      )}
    </div>
  );
}

function RunDetail({ id, onGone }) {
  const { t, lang, confirm } = useApp();
  const { data, error, reload } = useLoad(() => api.call("rag_run_result", { run: id }), [id]);
  const [open, setOpen] = useState(null);
  const [onlyFailed, setOnlyFailed] = useState(false);
  const [busy, run] = useBusy();
  const active = !data || ACTIVE.has(data.run.state);
  usePoll(reload, 3000, active);
  if (error && !data) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const r = data.run;
  const s = data.summary && data.summary.cases !== undefined ? data.summary : null;
  const cancel = () => run("cancel", async () => { await api.call("rag_run_cancel", { run: id }); reload(); });
  const remove = () => run("delete", async () => {
    if (!(await confirm({ title: t("rag_run_delete"), message: r.label, confirmLabel: t("delete"), danger: true }))) return;
    await api.call("rag_run_delete", { run: id, confirm: true });
    onGone();
  });
  const rows = (data.cases || []).filter((c) => !onlyFailed || !c.ok);
  const ks = s ? Object.keys(s.retrieval.recall) : [];
  const delta = (d) => (d ? <>{num(d.before, 2, lang)} → {num(d.after, 2, lang)} <span className="help">({signed(d.delta, lang)})</span></> : "—");
  return (
    <div className="space-y-4" data-run={id}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ margin: 0 }}>{r.label}</h2>
        <StateChip state={r.state} />
        {ACTIVE.has(r.state) && <Busy className="btn" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("cancel")}</Busy>}
        {!ACTIVE.has(r.state) && <Busy className="btn btn-danger" busy={busy.delete} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("delete")}</Busy>}
      </div>
      {ACTIVE.has(r.state) && <div className="panel"><Progress done={r.done} total={r.total} /><div className="help">{r.done} / {r.total}</div></div>}
      {r.error ? <div className="banner banner-danger">{r.error}</div> : null}
      {s && (
        <>
          {s.warnings.map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(270px, 1fr))" }}>
            <div className="panel space-y-1" data-piece="total">
              <b>{t("rag_pass_rate")}</b>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{share(s.pass_rate, lang)}</div>
              <div className="help">{t("rag_passed_of", { n: s.passed, total: s.cases })}{s.pass_ci95 ? ` · ${share(s.pass_ci95[0], lang)}–${share(s.pass_ci95[1], lang)}` : ""}{s.errors ? ` · ${t("rag_errors", { n: s.errors })}` : ""}</div>
            </div>
            <Attribution a={s.attribution} />
          </div>
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(270px, 1fr))" }}>
            <div className="panel space-y-1" data-piece="retriever">
              <b>{t("rag_piece_retriever")}</b> <span className="help">{t("rag_piece_retriever_q")}</span>
              <table className="tbl"><tbody>
                {ks.map((k) => <tr key={k}><td>recall@{k}</td><td className="num">{share(s.retrieval.recall[k], lang)}</td><td>nDCG@{k}</td><td className="num">{num(s.retrieval.ndcg[k], 2, lang)}</td></tr>)}
                <tr><td>MRR</td><td className="num">{num(s.retrieval.mrr, 2, lang)}</td><td>{t("rag_hit_rate")}</td><td className="num">{share(s.retrieval.hit_rate, lang)}</td></tr>
              </tbody></table>
              <div className="help">{t("rag_retrieval_n", { n: s.retrieval.cases })}</div>
            </div>
            <div className="panel space-y-1" data-piece="ranking">
              <b>{t("rag_piece_ranking")}</b> <span className="help">{t("rag_piece_ranking_q")}</span>
              {!s.rerank.applied && <div className="help">{t("rag_no_reranker")}</div>}
              <table className="tbl"><tbody>
                <tr><td>MRR</td><td className="num">{delta(s.rerank.mrr)}</td></tr>
                <tr><td>nDCG@{t("rag_cut")}</td><td className="num">{delta(s.rerank.ndcg)}</td></tr>
                <tr><td>recall@{t("rag_cut")}</td><td className="num">{delta(s.rerank.recall)}</td></tr>
              </tbody></table>
              <div className="help">{t("rag_rescued_demoted", { rescued: s.rerank.rescued, demoted: s.rerank.demoted })}</div>
            </div>
            <div className="panel space-y-1" data-piece="generator">
              <b>{t("rag_piece_generator")}</b> <span className="help">{t("rag_piece_generator_q")}</span>
              <table className="tbl"><tbody>
                <tr><td>{t("rag_faithfulness")}</td><td className="num">{share(s.generation.faithfulness, lang)}</td></tr>
                <tr><td>{t("rag_coverage")}</td><td className="num">{share(s.generation.coverage, lang)}</td></tr>
                <tr><td>{t("rag_right_with_passage")}</td><td className="num">{share(s.generation.with_passage.rate, lang)} <span className="help">({s.generation.with_passage.right}/{s.generation.with_passage.cases})</span></td></tr>
                <tr><td>{t("rag_abstain_without")}</td><td className="num">{share(s.generation.abstention.without_passage.rate, lang)} <span className="help">({s.generation.abstention.without_passage.abstained}/{s.generation.abstention.without_passage.cases})</span></td></tr>
                <tr><td>{t("rag_abstain_unanswerable")}</td><td className="num">{share(s.generation.abstention.unanswerable.rate, lang)} <span className="help">({s.generation.abstention.unanswerable.abstained}/{s.generation.abstention.unanswerable.cases})</span></td></tr>
              </tbody></table>
              <div className="help">{t("rag_judged_by")}: {Object.entries(s.generation.judge_modes).map(([m, n]) => `${t(`rag_judge_mode_${m}`)} ${n}`).join(" · ")}{s.generation.judge ? ` · ${s.generation.judge}` : ""}</div>
            </div>
          </div>
        </>
      )}
      <Section title={t("rag_per_case")} count={rows.length} actions={<label className="flex items-center gap-2 help"><input type="checkbox" checked={onlyFailed} onChange={(e) => setOnlyFailed(e.target.checked)} />{t("rag_only_failed")}</label>}>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table className="tbl">
            <thead><tr><th>{t("rag_question")}</th><th>{t("rag_result")}</th><th>{t("rag_first_rank")}</th><th>{t("rag_faithfulness")}</th><th>{t("rag_coverage")}</th><th>{t("rag_answer")}</th></tr></thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.case} data-case={c.case} data-piece={c.piece || "ok"} onClick={() => setOpen(c.case)} style={{ cursor: "pointer" }}>
                  <td style={{ minWidth: 220 }}>{c.question}{!c.answerable ? <> <Chip>{t("rag_unanswerable")}</Chip></> : null}</td>
                  <td>{c.ok ? <Chip className="chip-ok">{t("rag_ok")}</Chip> : <><Chip className={PIECE_CHIP[c.piece] || "chip-danger"}>{t(`rag_piece_${c.piece || "unknown"}`)}</Chip> <span className="help">{c.cause ? t(`rag_cause_${c.cause}`) : ""}</span></>}</td>
                  <td className="num">{c.first_rank ?? "—"}</td>
                  <td className="num">{share(c.faithfulness, lang)}</td>
                  <td className="num">{share(c.coverage, lang)}</td>
                  <td className="help" style={{ minWidth: 200 }}>{c.error || c.answer}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      {open && <CaseTrace run={id} caseId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Compare({ runs }) {
  const { t, lang } = useApp();
  const done = runs.filter((r) => !ACTIVE.has(r.state) && r.head && r.head.cases);
  const [a, setA] = useState(done[1]?.id || "");
  const [b, setB] = useState(done[0]?.id || "");
  const [data, setData] = useState(null);
  const [busy, run] = useBusy();
  const go = () => run("compare", async () => setData(await api.call("rag_compare", { a, b })));
  const pick = (value, onChange) => (
    <select className="field" value={value} onChange={(e) => onChange(e.target.value)} style={{ maxWidth: 380 }}>
      <option value="">—</option>
      {done.map((r) => <option key={r.id} value={r.id}>{r.label} · {share(r.head.pass_rate, lang)}</option>)}
    </select>
  );
  if (done.length < 2) return <Empty>{t("rag_compare_need")}</Empty>;
  return (
    <div className="space-y-4">
      <div className="panel flex flex-wrap items-end gap-3">
        <Field label={t("rag_compare_a")}>{pick(a, setA)}</Field>
        <Field label={t("rag_compare_b")}>{pick(b, setB)}</Field>
        <Busy className="btn btn-primary" busy={busy.compare} disabled={!a || !b || a === b} onClick={go}>{t("rag_compare_go")}</Busy>
      </div>
      {data && (
        <div className="space-y-4" data-compare={data.verdict}>
          <div className="panel space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Chip className={data.verdict.startsWith("b_better") ? "chip-ok" : data.verdict.startsWith("a_better") ? "chip-danger" : "chip-amber"}>{t(`rag_verdict_${data.verdict}`)}</Chip>
              <span>{t("rag_compare_counts", { shared: data.shared, improved: data.improved, regressed: data.regressed })}</span>
              {data.sign_test_p !== null && <span className="help">p = {num(data.sign_test_p, 3, lang)}</span>}
            </div>
            {Object.keys(data.differs).length > 0 && (
              <div className="help">{t("rag_differs")}: {Object.entries(data.differs).map(([k, v]) => `${k}: ${v.a ?? "—"} → ${v.b ?? "—"}`).join(" · ")}</div>
            )}
          </div>
          <div className="panel" style={{ overflowX: "auto" }}>
            <table className="tbl">
              <thead><tr><th>{t("rag_measure")}</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
              <tbody>
                {Object.entries(data.pieces).map(([k, v]) => (
                  <tr key={k}><td>{t(`rag_measure_${k.replace(/_at_\d+$/, "_at_k")}`)}{/_at_\d+$/.test(k) ? ` (${k.match(/\d+$/)[0]})` : ""}</td><td className="num">{num(v.a, 2, lang)}</td><td className="num">{num(v.b, 2, lang)}</td>
                    <td className="num">{v.delta === null ? "—" : <Chip className={v.delta > 0.02 ? "chip-ok" : v.delta < -0.02 ? "chip-danger" : ""}>{signed(v.delta, lang)}</Chip>}</td></tr>
                ))}
                {PIECES.filter((p) => (data.attribution.a?.[p] || 0) + (data.attribution.b?.[p] || 0) > 0).map((p) => (
                  <tr key={p}><td>{t("rag_failures_of")} {t(`rag_piece_${p}`)}</td><td className="num">{data.attribution.a?.[p] ?? 0}</td><td className="num">{data.attribution.b?.[p] ?? 0}</td><td /></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel" style={{ overflowX: "auto" }}>
            <table className="tbl">
              <thead><tr><th>{t("rag_question")}</th><th>A</th><th>B</th><th>{t("rag_change")}</th></tr></thead>
              <tbody>
                {data.cases.map((c) => (
                  <tr key={c.case} data-change={c.change}>
                    <td style={{ minWidth: 220 }}>{c.question}</td>
                    {[c.a, c.b].map((x, i) => <td key={i}>{x.ok ? <Chip className="chip-ok">{t("rag_ok")}</Chip> : <Chip className={PIECE_CHIP[x.piece] || "chip-danger"}>{t(`rag_piece_${x.piece || "unknown"}`)}</Chip>}</td>)}
                    <td><Chip className={CHANGE_CHIP[c.change]}>{t(`rag_change_${c.change}`)}</Chip></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ one set
function SetDetail({ id, onGone, onChanged }) {
  const { t, confirm } = useApp();
  const servers = useServers();
  const loaded = useLoad(() => api.call("rag_set_get", { set: id }), [id]);
  const { data, error } = loaded;
  const reload = () => { loaded.reload(); onChanged(); };      // the list of sets above shows the same counts
  const [tab, setTab] = useState("cases");
  const [runId, setRunId] = useState(null);
  const [busy, run] = useBusy();
  const anyActive = (data?.runs || []).some((r) => ACTIVE.has(r.state));
  usePoll(loaded.reload, 4000, anyActive);
  if (error && !data) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const remove = () => run("delete", async () => {
    if (!(await confirm({ title: t("rag_set_delete"), message: t("rag_set_delete_confirm", { name: data.set.name }), confirmLabel: t("delete"), danger: true }))) return;
    await api.call("rag_set_delete", { set: id, confirm: true });
    onGone();
  });
  const tabs = [{ id: "cases", label: `${t("rag_cases")} (${data.cases.length})` }, { id: "evaluate", label: t("rag_tab_evaluate") },
    { id: "results", label: `${t("rag_tab_results")} (${data.runs.length})` }, { id: "compare", label: t("rag_tab_compare") }];
  return (
    <div className="space-y-4" data-set={id}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ margin: 0 }}>{data.set.name}</h2>
        <span className="help">{t("rag_set_docs", { n: data.set.documents.length, chars: data.set.chars })}</span>
        <Busy className="btn btn-danger" busy={busy.delete} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("rag_set_delete")}</Busy>
      </div>
      <Tabs tabs={tabs} value={tab} onChange={setTab} />
      {tab === "cases" && <Cases detail={data} servers={servers} reload={reload} />}
      {tab === "evaluate" && <Evaluate detail={data} servers={servers} onStarted={(rid) => { setRunId(rid); setTab("results"); reload(); }} />}
      {tab === "results" && (
        <div className="space-y-4">
          {!data.runs.length && <Empty>{t("rag_no_runs")}</Empty>}
          <div className="space-y-2">
            {data.runs.map((r) => (
              <button key={r.id} type="button" className="panel flex flex-wrap items-center gap-2" data-run-card={r.id} onClick={() => setRunId(r.id)}
                style={{ textAlign: "left", width: "100%", outline: runId === r.id ? "2px solid var(--accent)" : "none" }}>
                <b className="mr-auto">{r.label}</b>
                <StateChip state={r.state} />
                {r.head?.cases ? <Chip className={r.head.pass_rate >= 0.8 ? "chip-ok" : r.head.pass_rate >= 0.5 ? "chip-amber" : "chip-danger"}>{r.head.passed}/{r.head.cases}</Chip> : null}
                {r.head?.main_piece ? <Chip className={PIECE_CHIP[r.head.main_piece]}>{t("rag_main_piece")}: {t(`rag_piece_${r.head.main_piece}`)}</Chip> : null}
                <span className="help"><Rel ts={r.created_ts} /></span>
              </button>
            ))}
          </div>
          {runId && <RunDetail key={runId} id={runId} onGone={() => { setRunId(null); reload(); }} />}
        </div>
      )}
      {tab === "compare" && <Compare runs={data.runs} />}
    </div>
  );
}

export default function Rag() {
  const { t, version } = useApp();
  const list = useLoad(() => api.call("rag_sets", {}), [version]);
  const [sel, setSel] = useState(null);
  const [creating, setCreating] = useState(false);
  const rows = list.data?.sets || [];
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("rag_title")}</h1>
        <button type="button" className="btn btn-primary" onClick={() => setCreating(!creating)}><Icon d={ICONS.plus} size={14} />{t("rag_new_set")}</button>
        <Busy className="btn" busy={false} onClick={list.reload}><Icon d={ICONS.refresh} size={14} />{t("reload")}</Busy>
      </div>
      <p className="help">{t("rag_help")}</p>
      <ErrorBox error={list.error} />
      {creating && <NewSet onCreated={(id) => { setCreating(false); setSel(id); list.reload(); }} />}
      <Section title={t("rag_sets")} count={rows.length}>
        {list.loading && !list.data && <Spinner />}
        {!list.loading && !rows.length && <Empty>{t("rag_no_sets")}</Empty>}
        <div className="space-y-2">
          {rows.map((r) => (
            <button key={r.id} type="button" className="panel flex flex-wrap items-center gap-2" data-set-card={r.id} onClick={() => setSel(r.id)}
              style={{ textAlign: "left", width: "100%", outline: sel === r.id ? "2px solid var(--accent)" : "none" }}>
              <b className="mr-auto">{r.name}</b>
              <span className="help">{t("rag_set_docs", { n: r.documents.length, chars: r.chars })}</span>
              <Chip className="chip-ok">{t("rag_state_accepted")}: {r.cases.accepted}</Chip>
              {r.cases.proposed > 0 && <Chip className="chip-amber">{t("rag_state_proposed")}: {r.cases.proposed}</Chip>}
              <Chip>{t("rag_runs_n", { n: r.runs })}</Chip>
            </button>
          ))}
        </div>
      </Section>
      {sel && <SetDetail key={sel} id={sel} onGone={() => { setSel(null); list.reload(); }} onChanged={list.reload} />}
    </div>
  );
}
