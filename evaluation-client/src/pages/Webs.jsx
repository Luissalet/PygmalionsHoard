import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { ACTIVE_STATES } from "../meta.js";
import { num } from "../format.js";

const pct = (v, lang) => (v == null ? "—" : `${num(v * 100, 1, lang)} %`);
const fmt = (v, lang, digits = 1) => (v == null ? "—" : num(v, digits, lang));
const money = (v, currency, lang) => (v == null ? "—" : `${num(v, v < 0.01 ? 5 : 4, lang)} ${currency}`);
const IMG = { width: "100%", borderRadius: 6, border: "1px solid var(--border)", background: "#fff" };

// ------------------------------------------------------------------ the form
function NewRun({ catalog }) {
  const { t, version, changed, notify } = useApp();
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const [sites, setSites] = useState(["ficha", "precios"]);
  const [url, setUrl] = useState("");
  const [models, setModels] = useState([]);
  const [prices, setPrices] = useState({});
  const [steps, setSteps] = useState("");
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [busy, run] = useBusy();
  const served = (modelsLoad.data?.models || []).filter((m) => !m.not_served);
  const toggle = (list, setList, id) => setList(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);
  const args = useMemo(() => {
    const a = { models, sites };
    if (url.trim()) a.urls = [{ url: url.trim() }];
    const priced = {};
    for (const id of models) {
      const name = served.find((m) => m.id === id)?.name;
      const p = prices[id];
      if (name && p && ((p.input !== "" && p.input != null) || (p.output !== "" && p.output != null))) priced[name] = { input: Number(p.input) || 0, output: Number(p.output) || 0 };
    }
    if (Object.keys(priced).length) a.prices = priced;
    if (steps !== "") a.settings = { max_steps: Number(steps) };
    return a;
  }, [models, sites, url, prices, steps, served]);
  const ready = models.length > 0 && (sites.length > 0 || url.trim().length > 8);
  useEffect(() => {
    if (!ready) { setPlan(null); setPlanError(null); return undefined; }
    let stale = false;
    const timer = setTimeout(() => {
      api.call("webrebuild_start", { ...args, dry_run: true }).then((p) => { if (!stale) { setPlan(p); setPlanError(null); } }).catch((e) => { if (!stale) { setPlan(null); setPlanError(e); } });
    }, 300);
    return () => { stale = true; clearTimeout(timer); };
  }, [args, ready]);
  const start = () => run("start", async () => {
    const r = await api.call("webrebuild_start", { ...args, wait_s: 0 });
    notify(t("wr_started"));
    changed();
    window.location.hash = `#/webs/${r.run.id}`;
  });
  const noBrowser = catalog && !catalog.browser?.found;
  return (
    <Section title={t("wr_new")}>
      <div className="panel space-y-4">
        <Field label={t("wr_sites")} hint={t("wr_sites_hint")}>
          <div className="grid gap-2 sm:grid-cols-2">
            {(catalog?.tasks || []).map((s) => (
              <label key={s.id} className="panel panel-tight flex gap-2" style={{ cursor: "pointer" }}>
                <input type="checkbox" checked={sites.includes(s.id)} onChange={() => toggle(sites, setSites, s.id)} />
                <span className="space-y-1" style={{ minWidth: 0 }}>
                  <b>{s.title}</b> {s.level && <Chip>{s.level}</Chip>}
                  <span className="help block" style={{ overflowWrap: "anywhere" }}>{s.summary}</span>
                  <span className="help block num">{t("wr_site_counts", { cases: s.cases, checkpoints: s.checkpoints, checks: s.checks })}</span>
                </span>
              </label>
            ))}
          </div>
        </Field>
        <Field label={t("wr_url")} hint={t("wr_url_hint")}><input className="field" type="url" placeholder="http://127.0.0.1:8080" value={url} onChange={(e) => setUrl(e.target.value)} /></Field>
        <Field label={t("bk_models")} hint={t("wr_models_hint")}>
          <div className="space-y-2">
            {modelsLoad.loading && !modelsLoad.data && <Spinner />}
            {served.map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2" style={{ minWidth: 220 }}>
                  <input type="checkbox" checked={models.includes(m.id)} onChange={() => toggle(models, setModels, m.id)} />
                  <span style={{ overflowWrap: "anywhere" }}>{m.name}</span>
                </label>
                {models.includes(m.id) && (
                  <span className="flex items-center gap-2 help">
                    {t("bk_price")}
                    <input className="field" style={{ width: 84 }} type="number" min="0" step="any" placeholder={t("bk_price_in")} aria-label={`${m.name}: ${t("bk_price_in")}`}
                      value={prices[m.id]?.input ?? ""} onChange={(e) => setPrices({ ...prices, [m.id]: { ...prices[m.id], input: e.target.value } })} />
                    <input className="field" style={{ width: 84 }} type="number" min="0" step="any" placeholder={t("bk_price_out")} aria-label={`${m.name}: ${t("bk_price_out")}`}
                      value={prices[m.id]?.output ?? ""} onChange={(e) => setPrices({ ...prices, [m.id]: { ...prices[m.id], output: e.target.value } })} />
                  </span>
                )}
              </div>
            ))}
            {modelsLoad.data && !served.length && <Empty>{t("bk_no_models")}</Empty>}
          </div>
        </Field>
        <Field label={t("wr_steps")} hint={t("wr_steps_hint")}><input className="field" style={{ maxWidth: 160 }} type="number" min="4" max="200" placeholder="30" value={steps} onChange={(e) => setSteps(e.target.value)} /></Field>
        <ErrorBox error={planError} />
        {noBrowser && <div className="banner banner-warn">{t("wr_no_browser")}</div>}
        {plan && (
          <div className="help space-y-1">
            {plan.notes.map((n, i) => <div key={i}>{t.msg(n)}</div>)}
          </div>
        )}
        <Busy className="btn btn-primary" busy={busy.start} disabled={!ready || !!planError || noBrowser} onClick={start}><Icon d={ICONS.run} size={14} />{t("wr_start")}</Busy>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ the public WebsiteBench tasks
function WebsiteBench() {
  const { t, lang, version, notify } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("webrebuild_websitebench", { action: "list" }), [version]);
  const [busy, run] = useBusy();
  const [open, setOpen] = useState(false);
  const doImport = (every) => run(every ? "all" : "index", async () => {
    const r = await api.call("webrebuild_websitebench", { action: "import", every });
    notify(t("wr_wb_imported", { n: r.imported.length, total: r.task_count }));
    reload();
  });
  const rows = data?.tasks || [];
  return (
    <Section title={t("wr_wb_title")} count={data?.imported ? data.task_count : undefined}>
      <div className="panel space-y-3">
        <p className="help">{t("wr_wb_help")}</p>
        <ErrorBox error={error} />
        {loading && !data && <Spinner />}
        <div className="flex flex-wrap gap-2">
          <Busy className="btn" busy={busy.index} onClick={() => doImport(false)}>{t(data?.imported ? "wr_wb_refresh" : "wr_wb_import")}</Busy>
          <Busy className="btn" busy={busy.all} onClick={() => doImport(true)}>{t("wr_wb_import_all")}</Busy>
          {rows.length > 0 && <button type="button" className="btn" aria-expanded={open} onClick={() => setOpen(!open)}>{t(open ? "wr_wb_hide" : "wr_wb_show", { n: rows.length })}</button>}
        </div>
        {data?.imported && <p className="help" style={{ overflowWrap: "anywhere" }}>{data.credit}</p>}
        {open && (
          <div className="scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>{t("wr_wb_task")}</th><th>{t("wr_wb_difficulty")}</th><th className="r">{t("wr_wb_cases")}</th><th>{t("wr_wb_kinds")}</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <td style={{ overflowWrap: "anywhere" }}>{r.id}{r.details?.description && <div className="help">{r.details.description}</div>}</td>
                    <td>{r.details?.difficulty || "—"}</td>
                    <td className="r num">{r.details ? num(r.details.cases, 0, lang) : "—"}</td>
                    <td className="help">{r.details ? Object.entries(r.details.by_kind || {}).map(([k, v]) => `${k} ${v}`).join(" · ") : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("webrebuild_list", { limit: 30 }), [version]);
  const rows = data?.runs || [];
  usePoll(reload, 3000, rows.some((r) => ACTIVE_STATES.has(r.state)));
  return (
    <Section title={t("history")} count={rows.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !rows.length && <Empty>{t("wr_none")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/webs/${r.id}`}>{t.msg(r.label)}</a><div className="help">{r.models.join(" · ")} — {r.tasks.join(", ")}</div></td>
                  <td><StateChip state={r.state} /></td>
                  <td style={{ minWidth: 120 }}><div className="help num">{r.done}/{r.total}</div><Progress done={r.done} total={r.total} /></td>
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

// ------------------------------------------------------------------ one rebuilt site: each case, its checks and its pictures side by side
function Checkpoint({ run, site, slot, cp }) {
  const { t, lang } = useApp();
  const [view, setView] = useState(cp.diff ? "diff" : "candidate");
  const pass = cp.score != null && cp.score >= cp.threshold;
  return (
    <div className="panel panel-tight space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <b style={{ overflowWrap: "anywhere" }}>{cp.name}</b>
        <span className="help">{cp.route} · {cp.width} px</span>
        <Chip className={pass ? "chip-ok" : "chip-danger"}>{t("wr_ssim")} {cp.score == null ? "—" : num(cp.score, 4, lang)}</Chip>
        {cp.regions?.length > 0 && <Chip>{t("wr_zones", { n: cp.regions.length })}</Chip>}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <figure className="space-y-1" style={{ margin: 0 }}>
          <figcaption className="help">{t("wr_reference")}</figcaption>
          {cp.reference ? <img src={api.webUrl(run, site, "reference", cp.reference)} alt={`${t("wr_reference")}: ${cp.name}`} loading="lazy" style={IMG} /> : <Empty>—</Empty>}
        </figure>
        <figure className="space-y-1" style={{ margin: 0 }}>
          <figcaption className="flex flex-wrap items-center gap-2 help">
            {t("wr_candidate")}
            {cp.diff && (
              <span role="group" aria-label={t("wr_view")} className="flex gap-1">
                <button type="button" className="btn btn-sm" aria-pressed={view === "diff"} onClick={() => setView("diff")}>{t("wr_view_diff")}</button>
                <button type="button" className="btn btn-sm" aria-pressed={view === "candidate"} onClick={() => setView("candidate")}>{t("wr_view_plain")}</button>
              </span>
            )}
          </figcaption>
          {cp.candidate ? <img src={api.webUrl(run, site, slot, view === "diff" && cp.diff ? cp.diff : cp.candidate)} alt={`${t("wr_candidate")}: ${cp.name}`} loading="lazy" style={IMG} />
            : <div className="banner banner-warn">{t("wr_no_picture")}</div>}
        </figure>
      </div>
    </div>
  );
}

function ItemDetail({ run, site, slot, threshold, onClose }) {
  const { t, lang, version } = useApp();
  const { data, error, loading } = useLoad(() => api.call("webrebuild_item", { run, site, model: slot }), [run, site, slot, version]);
  const [onlyFailed, setOnlyFailed] = useState(true);
  const cases = (data?.case_results || []).filter((c) => !onlyFailed || !c.passed);
  return (
    <div className="panel space-y-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{data?.model_name || slot} · {site}</h2>
        <label className="flex items-center gap-2 help"><input type="checkbox" checked={onlyFailed} onChange={(e) => setOnlyFailed(e.target.checked)} />{t("wr_only_failed")}</label>
        <button type="button" className="btn btn-sm" onClick={onClose}><Icon d={ICONS.close} size={14} />{t("close")}</button>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && (
        <>
          {data.outcome === "none" && <div className="banner banner-warn">{t(`wr_reason_${data.reason}`)} {t.msg(data.detail)}</div>}
          {data.reason === "no_site" && <div className="banner banner-warn">{t.msg(data.detail)}</div>}
          <div className="kpis">
            <div className="kpi"><b>{pct(data.score, lang)}</b><span>{t("wr_score")}</span></div>
            <div className="kpi"><b className="num">{data.passed ?? "—"}/{data.cases ?? "—"}</b><span>{t("wr_cases_passed")}</span></div>
            <div className="kpi"><b>{fmt(data.visual_mean, lang, 3)}</b><span>{t("wr_visual")}</span></div>
            <div className="kpi"><b>{pct(data.functional_rate, lang)}</b><span>{t("wr_functional")}</span></div>
            <div className="kpi"><b>{data.steps ?? "—"}</b><span>{t("wr_steps_used")}</span></div>
            <div className="kpi"><b>{data.build_s != null ? `${fmt(data.build_s, lang)} s` : "—"}</b><span>{t("wr_build_time")}</span></div>
          </div>
          {data.note && <p className="help">{t("wr_model_note")}: {data.note}</p>}
          {data.files?.length > 0 && <p className="help" style={{ overflowWrap: "anywhere" }}>{t("wr_files")}: {data.files.join(", ")}</p>}
          {!cases.length && <Empty>{t("wr_all_passed")}</Empty>}
          {cases.map((c) => (
            <div key={`${c.case}-${c.width}`} className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h3 style={{ overflowWrap: "anywhere" }}>{c.case}</h3><Chip>{c.width} px</Chip>
                <Chip className={c.passed ? "chip-ok" : "chip-danger"}>{t(c.passed ? "wr_case_pass" : "wr_case_fail")}</Chip>
                <span className="help num">{t("wr_checks_count", { ok: c.checks_passed, n: c.checks })}</span>
              </div>
              {c.failed_checks?.length > 0 && (
                <ul className="help">{c.failed_checks.map((f, i) => <li key={i} style={{ overflowWrap: "anywhere" }}>{t("wr_step")} {f.step} · {f.kind}: {f.detail}</li>)}</ul>
              )}
              {c.checkpoints.map((cp) => <Checkpoint key={cp.name} run={run} site={site} slot={slot} cp={{ ...cp, threshold }} />)}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ one run
function RunDetail({ id }) {
  const { t, lang, version, changed, notify } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("webrebuild_result", { run: id }), [id, version]);
  const run = data?.run;
  usePoll(reload, 2500, !!run && ACTIVE_STATES.has(run.state));
  const [pick, setPick] = useState(null);
  const [busy, doRun] = useBusy();
  if (!data) return <div className="space-y-3"><a href="#/webs" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_webs")}</a><ErrorBox error={error} />{loading && !error && <Spinner />}</div>;
  const s = data.summary;
  const active = ACTIVE_STATES.has(run.state);
  const cancel = () => doRun("cancel", async () => { await api.call("webrebuild_cancel", { run: id }); notify(t("wr_cancelled")); changed(); reload(); });
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/webs" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_webs")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{t.msg(run.label)}</h1>
        <StateChip state={run.state} />
        {active && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("wr_cancel")}</Busy>}
      </div>
      <ErrorBox error={error} />
      {run.error && <Failure m={run.error} />}
      <div className="panel space-y-2">
        <Progress done={run.done} total={run.total} />
        <div className="flex flex-wrap gap-4 help num">
          <span>{run.done}/{run.total}</span>
          <span>{t(`wr_phase_${run.progress.phase || "queued"}`)}</span>
          {run.progress.current && active && <span>{t("bk_now")}: {run.progress.current}</span>}
          <span>{t("wr_threshold")}: {num(run.threshold, 2, lang)} · {run.widths.join(" / ")} px</span>
        </div>
        {(run.progress.warnings || []).map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
      </div>
      <Section title={t("bk_summary")}>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("bk_model")}</th><th className="r">{t("wr_score")}</th><th className="r">{t("wr_cases_passed")}</th><th className="r">{t("wr_visual")}</th><th className="r">{t("wr_functional")}</th>
              <th className="r">{t("bk_tokens_in")}</th><th className="r">{t("bk_tokens_out")}</th><th className="r">{t("bk_cost_total")}</th><th className="r">{t("wr_build_time")}</th></tr></thead>
            <tbody>
              {s.models.map((m) => (
                <tr key={m.slot}>
                  <td style={{ overflowWrap: "anywhere" }}>{m.model}
                    <div className="flex flex-wrap gap-1">
                      {s.best === m.model && <Chip className="chip-ok">{t("bk_best")}</Chip>}
                      {s.fastest === m.model && <Chip className="chip-info">{t("bk_fastest")}</Chip>}
                      {s.cheapest === m.model && <Chip className="chip-accent">{t("bk_cheapest")}</Chip>}
                      {m.none > 0 && <Chip>{t("wr_none_count", { n: m.none })}</Chip>}
                    </div></td>
                  <td className="r num">{pct(m.score, lang)}</td>
                  <td className="r num">{m.cases_passed}/{m.cases}</td>
                  <td className="r num">{fmt(m.visual_mean, lang, 3)}</td>
                  <td className="r num">{pct(m.functional_rate, lang)}</td>
                  <td className="r num">{fmt(m.tokens_in, lang, 0)}</td>
                  <td className="r num">{fmt(m.tokens_out, lang, 0)}</td>
                  <td className="r num">{money(m.cost, s.currency, lang)}</td>
                  <td className="r num">{m.build_s != null ? `${fmt(m.build_s, lang)} s` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="help">{t("wr_legend")}</p>
      </Section>
      {data.tasks.map((task) => {
        const shown = pick && pick.site === task.id ? pick : null;
        return (
          <Section key={task.id} title={task.title} count={task.rows.length}>
            <p className="help">{t("wr_reference_valid", { valid: task.reference_valid ?? "—", total: task.reference_cases })}{task.url ? ` · ${task.url}` : ""}</p>
            <div className="panel scroll-x" style={{ padding: 0 }}>
              <table className="tbl">
                <thead><tr><th>{t("bk_model")}</th><th>{t("status")}</th><th className="r">{t("wr_score")}</th><th className="r">{t("wr_cases_passed")}</th><th className="r">{t("wr_visual")}</th>
                  <th className="r">{t("wr_functional")}</th><th className="r">{t("wr_steps_used")}</th><th className="r">{t("bk_cost")}</th><th className="r">{t("wr_build_time")}</th><th /></tr></thead>
                <tbody>
                  {task.rows.map((r) => (
                    <tr key={r.slot}>
                      <td style={{ overflowWrap: "anywhere" }}>{r.model_name}</td>
                      <td>{r.outcome === "none" ? <Chip>{t(`wr_reason_${r.reason || "none"}`)}</Chip> : <StateChip state={r.state === "done" ? "done" : "running"} />}</td>
                      <td className="r num"><b>{pct(r.score, lang)}</b></td>
                      <td className="r num">{r.passed ?? "—"}/{r.cases ?? "—"}</td>
                      <td className="r num">{fmt(r.visual_mean, lang, 3)}</td>
                      <td className="r num">{pct(r.functional_rate, lang)}</td>
                      <td className="r num">{r.steps ?? "—"}</td>
                      <td className="r num">{money(r.cost, s.currency, lang)}</td>
                      <td className="r num">{r.build_s != null ? `${fmt(r.build_s, lang)} s` : "—"}</td>
                      <td>{r.state === "done" && r.outcome !== "none" && (
                        <button type="button" className="btn btn-sm" aria-expanded={shown?.slot === r.slot} onClick={() => setPick(shown?.slot === r.slot ? null : { site: task.id, slot: r.slot })}>{t("wr_open")}</button>)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {shown && <ItemDetail key={`${shown.site}-${shown.slot}`} run={id} site={shown.site} slot={shown.slot} threshold={run.threshold} onClose={() => setPick(null)} />}
          </Section>
        );
      })}
    </div>
  );
}

export default function Webs({ param }) {
  const { t, version } = useApp();
  const catalog = useLoad(() => api.call("webrebuild_catalog", {}), [version]);
  if (param) return <RunDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_webs")}</h1>
      <p className="help">{t("wr_help")}</p>
      <NewRun catalog={catalog.data} />
      <History />
      <WebsiteBench />
    </div>
  );
}
