import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { ACTIVE_STATES } from "../meta.js";
import { duration, num, pct } from "../format.js";

const DIMENSIONS = ["style_fidelity", "concept_wow", "motion_craft", "design_typography", "finish_texture", "technical"];
const OUTCOME_CHIP = { done: "chip-ok", unrendered: "chip-danger", none: "" };
const PHASES = ["writing", "rendering", "judging"];
const fileUrl = api.videoUrl;
const fmt = (v, lang) => (v == null ? "—" : num(v, 1, lang));

// ------------------------------------------------------------------ the briefs
function Briefs({ catalog }) {
  const { t } = useApp();
  const [category, setCategory] = useState("");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(null);
  const list = useLoad(() => api.call("videobench_catalog", { category, query, standalone: true, limit: 12 }), [category, query]);
  const full = useLoad(() => (open ? api.call("videobench_prompt", { prompt: open }) : Promise.resolve(null)), [open]);
  const rows = list.data?.prompts || [];
  return (
    <Section title={t("vb_briefs")} count={list.data?.matching}>
      <div className="panel space-y-3">
        <p className="help">{t("vb_briefs_help", { total: catalog?.total ?? "…", standalone: catalog?.standalone ?? "…" })}</p>
        <div className="flex flex-wrap gap-3">
          <Field label={t("vb_category")}>
            <select className="field" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">{t("vb_all")}</option>
              {Object.keys(catalog?.categories || {}).map((c) => <option key={c} value={c}>{t(`vb_cat_${c}`)} ({catalog.categories[c]})</option>)}
            </select>
          </Field>
          <Field label={t("search")}><input className="field" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("vb_search_ph")} /></Field>
        </div>
        <ErrorBox error={list.error} />
        {list.loading && !list.data && <Spinner />}
        <div className="space-y-2">
          {rows.map((p) => (
            <div key={p.id} className="panel panel-tight space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <b className="mono" style={{ overflowWrap: "anywhere" }}>{p.id}</b>
                <Chip>{t(`vb_cat_${p.category}`)}</Chip>
                {p.tech_tags.map((x) => <Chip key={x}>{x}</Chip>)}
                <span className="help ml-auto">{t("vb_by")} <a href={p.author_url} target="_blank" rel="noreferrer">{p.author}</a> · <a href={p.post_url} target="_blank" rel="noreferrer">{t("vb_post")}</a></span>
              </div>
              <div className="help" style={{ overflowWrap: "anywhere" }}>{p.excerpt}…</div>
              <button type="button" className="btn btn-sm" onClick={() => setOpen(open === p.id ? null : p.id)}>{open === p.id ? t("close") : t("vb_whole_prompt")}</button>
              {open === p.id && <pre className="pre">{full.data?.prompt || "…"}</pre>}
            </div>
          ))}
        </div>
        {list.data && list.data.matching > rows.length && <div className="help">{t("vb_more", { n: list.data.matching - rows.length })}</div>}
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ the form
function NewRun({ catalog }) {
  const { t, version, changed, notify } = useApp();
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const [model, setModel] = useState("");
  const [jury, setJury] = useState("");
  const [form, setForm] = useState({ count: 10, category: "", standalone: true, seed: 0, max_tokens: "", frames: "" });
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [busy, run] = useBusy();
  const models = (modelsLoad.data?.models || []).filter((m) => !m.not_served);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));
  const args = useMemo(() => {
    const a = { model, count: Math.max(1, Number(form.count) || 1), seed: Number(form.seed) || 0, standalone: form.standalone ? true : null };
    if (jury) a.jury = jury;
    if (form.category) a.category = form.category;
    const settings = {};
    if (form.max_tokens !== "") settings.max_tokens = Number(form.max_tokens);
    if (form.frames !== "") settings.frames = Number(form.frames);
    if (Object.keys(settings).length) a.settings = settings;
    return a;
  }, [model, jury, form]);
  useEffect(() => {
    if (!model) { setPlan(null); setPlanError(null); return undefined; }
    let stale = false;
    const timer = setTimeout(() => {
      api.call("videobench_start", { ...args, dry_run: true }).then((p) => { if (!stale) { setPlan(p); setPlanError(null); } }).catch((e) => { if (!stale) { setPlan(null); setPlanError(e); } });
    }, 250);
    return () => { stale = true; clearTimeout(timer); };
  }, [args, model]);
  const start = () => run("start", async () => {
    const r = await api.call("videobench_start", { ...args, wait_s: 0 });
    notify(t("vb_started"));
    changed();
    window.location.hash = `#/videos/${r.run.id}`;
  });
  const choose = (value, setter, empty) => (
    <select className="field" value={value} onChange={(e) => setter(e.target.value)}>
      <option value="">{empty}</option>
      {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
    </select>
  );
  return (
    <Section title={t("vb_new")}>
      <div className="panel space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("vb_writer")} hint={t("vb_writer_hint")}>{choose(model, setModel, t("vb_choose_model"))}</Field>
          <Field label={t("vb_jury")} hint={t("vb_jury_hint")}>{choose(jury, setJury, t("vb_same_model"))}</Field>
          <Field label={t("vb_count")} hint={t("vb_count_hint")}><input className="field" type="number" min="1" max="200" value={form.count} onChange={set("count")} /></Field>
          <Field label={t("vb_category")}>
            <select className="field" value={form.category} onChange={set("category")}>
              <option value="">{t("vb_all")}</option>
              {Object.keys(catalog?.categories || {}).map((c) => <option key={c} value={c}>{t(`vb_cat_${c}`)}</option>)}
            </select>
          </Field>
          <Field label={t("vb_seed")} hint={t("vb_seed_hint")}><input className="field" type="number" min="0" value={form.seed} onChange={set("seed")} /></Field>
          <Field label={t("max_tokens")} hint={t("vb_tokens_hint")}><input className="field" type="number" min="16" placeholder="16000" value={form.max_tokens} onChange={set("max_tokens")} /></Field>
        </div>
        <label className="flex items-center gap-2"><input type="checkbox" checked={form.standalone} onChange={set("standalone")} />{t("vb_standalone")}</label>
        <ErrorBox error={planError} />
        {catalog && !catalog.browser.found && <div className="banner banner-warn">{t("vb_no_browser")}</div>}
        {plan && (
          <div className="help">
            {t("vb_plan", { n: plan.n, writer: plan.model, jury: plan.jury })} {t(plan.jury_mode === "images" ? "vb_jury_images" : "vb_jury_text")}
            {plan.same_model_judges ? ` ${t("vb_same_warn")}` : ""}
          </div>
        )}
        <Busy className="btn btn-primary" busy={busy.start} disabled={!model || !!planError || (catalog && !catalog.browser.found)} onClick={start}><Icon d={ICONS.run} size={14} />{t("vb_start")}</Busy>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, lang, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("videobench_list", { limit: 30 }), [version]);
  const rows = data?.runs || [];
  usePoll(reload, 3000, rows.some((r) => ACTIVE_STATES.has(r.state)));
  return (
    <Section title={t("history")} count={rows.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !rows.length && <Empty>{t("vb_none")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th className="r">{t("vb_mean")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/videos/${r.id}`}>{t.msg(r.label)}</a><div className="help">{r.model.name}{r.same_model_judges ? "" : ` · ${t("vb_jury")}: ${r.jury.name}`}</div></td>
                  <td><StateChip state={r.state} /></td>
                  <td style={{ minWidth: 120 }}><div className="help num">{r.done}/{r.total}</div><Progress done={r.done} total={r.total} /></td>
                  <td className="r num">{fmt(r.summary?.mean_score, lang)}</td>
                  <td className="help"><Rel ts={r.created_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ one item: the frames, the page to play, the jury
function Player({ run, prompt, duration: length }) {
  const { t, lang } = useApp();
  const frame = useRef(null);
  const [at, setAt] = useState(0);
  const [round, setRound] = useState(0);
  const max = Math.max(1, Number(length) || 10);
  const seek = (value) => { setAt(value); frame.current?.contentWindow?.postMessage({ seek: Number(value) }, "*"); };
  return (
    <div className="space-y-2">
      <iframe key={round} ref={frame} title={t("vb_preview")} src={fileUrl(run, prompt, "play.html")} sandbox="allow-scripts" style={{ width: "100%", aspectRatio: "16 / 9", border: "1px solid var(--border)", borderRadius: 8, background: "#000" }} />
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className="btn btn-sm" onClick={() => { setAt(0); setRound((x) => x + 1); }}><Icon d={ICONS.run} size={14} />{t("vb_play")}</button>
        <input type="range" min="0" max={max} step="0.05" value={at} onChange={(e) => seek(e.target.value)} aria-label={t("vb_scrub")} style={{ flex: 1, minWidth: 160 }} />
        <span className="help num">{num(at, 2, lang)} / {num(max, 1, lang)} s</span>
      </div>
    </div>
  );
}

function ItemDetail({ run, prompt, onClose }) {
  const { t, lang, version } = useApp();
  const { data, error, loading } = useLoad(() => api.call("videobench_item", { run, prompt }), [run, prompt, version]);
  const jury = data?.jury;
  const auto = data?.auto || {};
  const render = data?.render || {};
  return (
    <div className="panel space-y-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{prompt}</h2>
        <button type="button" className="btn btn-sm" onClick={onClose}><Icon d={ICONS.close} size={14} />{t("close")}</button>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && (
        <>
          <div className="help">{t("vb_by")} <a href={data.brief.author_url} target="_blank" rel="noreferrer">{data.brief.author}</a> · <a href={data.brief.post_url} target="_blank" rel="noreferrer">{t("vb_post")}</a> · {t(`vb_cat_${data.brief.category}`)}</div>
          {data.outcome && data.outcome !== "done" && <div className="banner banner-warn"><Chip className={OUTCOME_CHIP[data.outcome]}>{t(`vb_out_${data.outcome}`)}</Chip> {data.reason && <Chip>{t(`vb_reason_${data.reason}`)}</Chip>} {t.msg(data.detail)}</div>}
          {data.files?.["sheet.png"] && <img src={fileUrl(run, prompt, "sheet.png")} alt={t("vb_sheet")} style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />}
          {data.files?.["index.html"] && data.files?.["sheet.png"] && <Player run={run} prompt={prompt} duration={render.duration} />}
          {data.files?.["sheet.png"] && (
            <div className="kpis">
              <div className="kpi"><b>{auto.moves ? t("yes") : t("no")}</b><span>{t("vb_moves")}</span></div>
              <div className="kpi"><b>{render.deterministic ? t("yes") : t("no")}</b><span>{t("vb_deterministic")}</span></div>
              <div className="kpi"><b>{(render.page_errors || []).length}</b><span>{t("vb_page_errors")}</span></div>
              <div className="kpi"><b>{(render.blocked_requests || []).length}</b><span>{t("vb_blocked")}</span></div>
              <div className="kpi"><b>{auto.blank_frames ?? "—"}</b><span>{t("vb_blank")}</span></div>
              <div className="kpi"><b>{render.duration != null ? `${num(render.duration, 1, lang)} s` : "—"}</b><span>{t("vb_length")}</span></div>
              <div className="kpi"><b>{data.html_chars != null ? num(data.html_chars / 1000, 1, lang) + " k" : "—"}</b><span>{t("vb_html_size")}</span></div>
              <div className="kpi"><b>{data.latency_ms != null ? duration(data.latency_ms) : "—"}</b><span>{t("vb_writing_time")}</span></div>
            </div>
          )}
          {jury?.scores && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2"><h3>{t("vb_jury")}</h3><Chip>{t(data.jury_mode === "images" ? "vb_jury_images_short" : "vb_jury_text_short")}</Chip><b className="num">{fmt(data.mean, lang)}</b></div>
              <div className="grid gap-2 sm:grid-cols-2">
                {DIMENSIONS.map((d) => (
                  <div key={d}>
                    <div className="flex justify-between"><span>{t(`vb_dim_${d}`)}</span><b className="num">{fmt(jury.scores[d], lang)}</b></div>
                    <Progress done={(jury.scores[d] || 0) * 10} total={100} />
                  </div>
                ))}
              </div>
              {jury.summary && <p>{jury.summary}</p>}
              {jury.critical?.length > 0 && <div className="banner banner-warn"><b>{t("vb_critical")}</b><ul>{jury.critical.map((c, i) => <li key={i}>{c}</li>)}</ul></div>}
              {jury.fixes?.length > 0 && (
                <div><b>{t("vb_fixes")}</b>
                  <ol style={{ paddingLeft: 18 }}>{jury.fixes.map((f, i) => <li key={i}><span className="mono">{f.t}</span> {f.problem} <span className="help">→ {f.fix}</span></li>)}</ol>
                </div>
              )}
            </div>
          )}
          <details><summary>{t("vb_brief")}</summary><pre className="pre">{data.brief.prompt}</pre></details>
          {data.files?.["index.html"] && <a className="btn btn-sm" href={fileUrl(run, prompt, "index.html")}>{t("vb_download")}</a>}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ one run
function RunDetail({ id }) {
  const { t, lang, version, changed, notify } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("videobench_result", { run: id }), [id, version]);
  const run = data?.run;
  usePoll(reload, 2500, !!run && ACTIVE_STATES.has(run.state));
  const [pick, setPick] = useState(null);
  const [busy, doRun] = useBusy();
  if (!data) return <div className="space-y-3"><a href="#/videos" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_videos")}</a><ErrorBox error={error} />{loading && !error && <Spinner />}</div>;
  const s = data.summary;
  const active = ACTIVE_STATES.has(run.state);
  const cancel = () => doRun("cancel", async () => { await api.call("videobench_cancel", { run: id }); notify(t("vb_cancelled")); changed(); reload(); });
  const copy = async () => {
    try { await navigator.clipboard.writeText(data.markdown); notify(t("hb_copied")); } catch { notify(t("hb_copy_failed"), "error"); }
  };
  const phase = run.progress.phase;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/videos" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_videos")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{t.msg(run.label)}</h1>
        <StateChip state={run.state} />
        <button type="button" className="btn" onClick={copy}><Icon d={ICONS.copy} size={14} />{t("hb_copy")}</button>
        {active && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("vb_cancel")}</Busy>}
      </div>
      <ErrorBox error={error} />
      {run.error && <Failure m={run.error} />}
      <div className="panel space-y-2">
        <Progress done={run.done} total={run.total} />
        <div className="flex flex-wrap gap-4 help num">
          <span>{run.done}/{run.total}</span>
          {PHASES.map((p) => <span key={p} style={{ fontWeight: phase === p ? 700 : 400 }}>{t(`vb_phase_${p}`)} {run.progress[{ writing: "written", rendering: "rendered", judging: "judged" }[p]] ?? 0}/{run.n}</span>)}
          <span>{t("vb_writer")}: {run.model.name}</span>
          <span>{t("vb_jury")}: {run.jury.name}{run.same_model_judges ? ` (${t("vb_same_short")})` : ""}</span>
        </div>
        {(run.progress.warnings || []).map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
      </div>
      <Section title={t("vb_summary")}>
        <div className="panel space-y-3">
          <div className="kpis">
            <div className="kpi"><b>{fmt(s.mean_score, lang)}</b><span>{t("vb_mean")}</span></div>
            <div className="kpi"><b>{s.judged}/{s.prompts}</b><span>{t("vb_judged")}</span></div>
            <div className="kpi"><b>{s.render_rate == null ? "—" : pct(s.render_rate, 0, lang)}</b><span>{t("vb_render_rate")}</span></div>
            <div className="kpi"><b>{s.usable_rate == null ? "—" : pct(s.usable_rate, 0, lang)}</b><span>{t("vb_usable_rate")}</span></div>
            <div className="kpi"><b>{s.deterministic_rate == null ? "—" : pct(s.deterministic_rate, 0, lang)}</b><span>{t("vb_deterministic")}</span></div>
            <div className="kpi"><b>{s.none}</b><span>{t("hb_none_n")}</span></div>
            <div className="kpi"><b>{s.tokens_out_mean != null ? num(s.tokens_out_mean, 0, lang) : "—"}</b><span>{t("hb_tokens_out")}</span></div>
            <div className="kpi"><b>{s.latency_s_mean != null ? `${num(s.latency_s_mean, 0, lang)} s` : "—"}</b><span>{t("vb_writing_time")}</span></div>
          </div>
          {s.by_dimension && s.judged > 0 && (
            <div className="grid gap-2 sm:grid-cols-2">
              {DIMENSIONS.map((d) => (
                <div key={d}>
                  <div className="flex justify-between"><span>{t(`vb_dim_${d}`)}</span><b className="num">{fmt(s.by_dimension[d], lang)}</b></div>
                  <Progress done={(s.by_dimension[d] || 0) * 10} total={100} />
                </div>
              ))}
            </div>
          )}
          <p className="help">{t("vb_legend")}{s.same_model_judges ? ` ${t("vb_same_warn")}` : ""}</p>
        </div>
      </Section>
      <Section title={t("vb_items")} count={data.items.length}>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th></th><th>{t("vb_brief")}</th><th>{t("status")}</th><th className="r">{t("vb_mean")}</th><th>{t("vb_moves")}</th></tr></thead>
            <tbody>
              {data.items.map((r) => (
                <tr key={r.prompt} onClick={() => setPick(pick === r.prompt ? null : r.prompt)} style={{ cursor: "pointer", background: pick === r.prompt ? "var(--hoard-elevated)" : undefined }}>
                  <td style={{ width: 110 }}>{r.has_sheet && <img src={fileUrl(id, r.prompt, "sheet.png")} alt="" loading="lazy" style={{ width: 100, borderRadius: 4 }} />}</td>
                  <td style={{ overflowWrap: "anywhere", minWidth: 180 }}><button type="button" className="btn btn-sm" aria-expanded={pick === r.prompt}>{r.prompt}</button><div className="help">{t(`vb_cat_${r.category}`)} · {r.author}</div></td>
                  <td><Chip className={OUTCOME_CHIP[r.outcome]}>{t(`vb_out_${r.outcome || "queued"}`)}</Chip>{r.reason && <div className="help">{t(`vb_reason_${r.reason}`)}</div>}</td>
                  <td className="r num">{fmt(r.mean, lang)}</td>
                  <td>{r.moves == null ? "—" : r.usable ? t("vb_usable") : r.moves ? t("vb_moves_ok") : t("no")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      {pick && <ItemDetail key={pick} run={id} prompt={pick} onClose={() => setPick(null)} />}
    </div>
  );
}

export default function Videos({ param }) {
  const { t, version } = useApp();
  const catalog = useLoad(() => api.call("videobench_catalog", { limit: 1 }), [version]);
  if (param) return <RunDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_videos")}</h1>
      <p className="help">{t("vb_help")}</p>
      <NewRun catalog={catalog.data} />
      <History />
      <Briefs catalog={catalog.data} />
    </div>
  );
}
