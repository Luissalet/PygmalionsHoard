import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { ACTIVE_STATES } from "../meta.js";
import { num } from "../format.js";

const DIMENSIONS = ["fidelity", "craft", "polish", "technical"];
const OUTCOME_CHIP = { done: "chip-ok", unrendered: "chip-danger", none: "" };
const fileUrl = api.briefUrl;
const fmt = (v, lang, digits = 1) => (v == null ? "—" : num(v, digits, lang));
const money = (v, currency, lang) => (v == null ? "—" : `${num(v, v < 0.01 ? 5 : 4, lang)} ${currency}`);
const SORTS = ["order", "score", "time", "cost"];

// ------------------------------------------------------------------ the form
function NewRun({ catalog }) {
  const { t, version, changed, notify } = useApp();
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const [source, setSource] = useState("catalog");
  const [picked, setPicked] = useState(["dragon-castle"]);
  const [own, setOwn] = useState({ prompt: "", title: "", kind: "page", criteria: "" });
  const [models, setModels] = useState([]);
  const [prices, setPrices] = useState({});
  const [judge, setJudge] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [busy, run] = useBusy();
  const served = (modelsLoad.data?.models || []).filter((m) => !m.not_served);
  const choices = useMemo(() => [...served.map((m) => ({ id: m.id, name: m.name })), ...(catalog?.cli_agents || []).map((n) => ({ id: `cli:${n}`, name: n, cli: true }))], [served, catalog]);
  const toggle = (list, setList, id) => setList(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);
  const args = useMemo(() => {
    const a = { models };
    if (source === "catalog") a.brief_ids = picked; else { a.brief = own.prompt; a.kind = own.kind; a.title = own.title; a.criteria = own.criteria; }
    const priced = {};
    for (const id of models) {
      const name = choices.find((c) => c.id === id)?.name;
      const p = prices[id];
      if (name && p && (p.input !== "" && p.input != null || p.output !== "" && p.output != null)) priced[name] = { input: Number(p.input) || 0, output: Number(p.output) || 0 };
    }
    if (Object.keys(priced).length) a.prices = priced;
    if (judge) a.judge = judge;
    if (maxTokens !== "") a.settings = { max_tokens: Number(maxTokens) };
    return a;
  }, [models, source, picked, own, prices, judge, maxTokens, choices]);
  const ready = models.length > 0 && (source === "catalog" ? picked.length > 0 : own.prompt.trim().length >= 10);
  useEffect(() => {
    if (!ready) { setPlan(null); setPlanError(null); return undefined; }
    let stale = false;
    const timer = setTimeout(() => {
      api.call("briefbench_start", { ...args, dry_run: true }).then((p) => { if (!stale) { setPlan(p); setPlanError(null); } }).catch((e) => { if (!stale) { setPlan(null); setPlanError(e); } });
    }, 300);
    return () => { stale = true; clearTimeout(timer); };
  }, [args, ready]);
  const start = () => run("start", async () => {
    const r = await api.call("briefbench_start", { ...args, wait_s: 0 });
    notify(t("bk_started"));
    changed();
    window.location.hash = `#/encargos/${r.run.id}`;
  });
  const needsBrowser = plan?.browser?.needed && !plan.browser.found;
  return (
    <Section title={t("bk_new")}>
      <div className="panel space-y-4">
        <div className="flex flex-wrap gap-4" role="radiogroup" aria-label={t("bk_source")}>
          <label className="flex items-center gap-2"><input type="radio" name="bk-source" checked={source === "catalog"} onChange={() => setSource("catalog")} />{t("bk_from_catalog")}</label>
          <label className="flex items-center gap-2"><input type="radio" name="bk-source" checked={source === "own"} onChange={() => setSource("own")} />{t("bk_write_own")}</label>
        </div>
        {source === "catalog" ? (
          <div className="grid gap-2 sm:grid-cols-2">
            {(catalog?.briefs || []).map((b) => (
              <label key={b.id} className="panel panel-tight flex gap-2" style={{ cursor: "pointer" }}>
                <input type="checkbox" checked={picked.includes(b.id)} onChange={() => toggle(picked, setPicked, b.id)} />
                <span className="space-y-1" style={{ minWidth: 0 }}>
                  <b>{b.title}</b> <Chip>{t(`bk_kind_${b.kind}`)}</Chip>
                  <span className="help block" style={{ overflowWrap: "anywhere" }}>{b.excerpt}…</span>
                </span>
              </label>
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            <Field label={t("bk_own_prompt")} hint={t("bk_own_prompt_hint")}><textarea className="field" rows={5} value={own.prompt} onChange={(e) => setOwn({ ...own, prompt: e.target.value })} /></Field>
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label={t("bk_own_title")}><input className="field" value={own.title} onChange={(e) => setOwn({ ...own, title: e.target.value })} /></Field>
              <Field label={t("bk_own_kind")}>
                <select className="field" value={own.kind} onChange={(e) => setOwn({ ...own, kind: e.target.value })}>
                  <option value="page">{t("bk_kind_page")}</option>
                  <option value="text">{t("bk_kind_text")}</option>
                </select>
              </Field>
              <Field label={t("bk_own_criteria")} hint={t("bk_own_criteria_hint")}><input className="field" value={own.criteria} onChange={(e) => setOwn({ ...own, criteria: e.target.value })} /></Field>
            </div>
          </div>
        )}
        <Field label={t("bk_models")} hint={t("bk_models_hint")}>
          <div className="space-y-2">
            {modelsLoad.loading && !modelsLoad.data && <Spinner />}
            {choices.map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2" style={{ minWidth: 220 }}>
                  <input type="checkbox" checked={models.includes(m.id)} onChange={() => toggle(models, setModels, m.id)} />
                  <span style={{ overflowWrap: "anywhere" }}>{m.name}</span>{m.cli && <Chip>{t("bk_cli")}</Chip>}
                </label>
                {models.includes(m.id) && !m.cli && (
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
            {modelsLoad.data && !choices.length && <Empty>{t("bk_no_models")}</Empty>}
          </div>
        </Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("bk_judge")} hint={t("bk_judge_hint")}>
            <select className="field" value={judge} onChange={(e) => setJudge(e.target.value)}>
              <option value="">{t("bk_no_judge")}</option>
              {served.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          <Field label={t("max_tokens")} hint={t("bk_tokens_hint")}><input className="field" type="number" min="16" placeholder="16000" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} /></Field>
        </div>
        <ErrorBox error={planError} />
        {needsBrowser && <div className="banner banner-warn">{t("bk_no_browser")}</div>}
        {plan && (
          <div className="help space-y-1">
            <div>{t("bk_plan", { models: plan.models.length, briefs: plan.briefs.length, calls: plan.calls.answers })}{plan.judge ? ` ${t("bk_plan_judge", { judge: plan.judge })}` : ""}</div>
            {plan.same_model_judges && <div>{t("bk_same_warn")}</div>}
            {plan.notes.slice(1).map((n, i) => <div key={i}>{t.msg(n)}</div>)}
          </div>
        )}
        <Busy className="btn btn-primary" busy={busy.start} disabled={!ready || !!planError || needsBrowser} onClick={start}><Icon d={ICONS.run} size={14} />{t("bk_start")}</Busy>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("briefbench_list", { limit: 30 }), [version]);
  const rows = data?.runs || [];
  usePoll(reload, 3000, rows.some((r) => ACTIVE_STATES.has(r.state)));
  return (
    <Section title={t("history")} count={rows.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !rows.length && <Empty>{t("bk_none")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/encargos/${r.id}`}>{t.msg(r.label)}</a><div className="help">{r.models.map((m) => m.name).join(" · ")}</div></td>
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

// ------------------------------------------------------------------ the answer of one model: its page to play, its pictures, its figures and what the judge said
function TextAnswer({ run, brief, slot }) {
  const [body, setBody] = useState(null);
  useEffect(() => {
    let stale = false;
    fetch(fileUrl(run, brief, slot, "answer.txt")).then((r) => (r.ok ? r.text() : "")).then((x) => { if (!stale) setBody(x); }).catch(() => { if (!stale) setBody(""); });
    return () => { stale = true; };
  }, [run, brief, slot]);
  if (body === null) return <Spinner />;
  return <pre className="pre" style={{ maxHeight: 360, overflow: "auto", whiteSpace: "pre-wrap" }}>{body}</pre>;
}

function ItemDetail({ run, brief, kind, slot, currency, onClose }) {
  const { t, lang, version } = useApp();
  const { data, error, loading } = useLoad(() => api.call("briefbench_item", { run, brief, model: slot }), [run, brief, slot, version]);
  const judge = data?.judge;
  const render = data?.render || {};
  const auto = data?.auto || {};
  return (
    <div className="panel space-y-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{data?.model_name || slot} · {data?.brief?.title || brief}</h2>
        <button type="button" className="btn btn-sm" onClick={onClose}><Icon d={ICONS.close} size={14} />{t("close")}</button>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && (
        <>
          {data.outcome && data.outcome !== "done" && <div className="banner banner-warn"><Chip className={OUTCOME_CHIP[data.outcome]}>{t(`bk_out_${data.outcome}`)}</Chip> {data.reason && <Chip>{t(`bk_reason_${data.reason}`)}</Chip>} {t.msg(data.detail)}</div>}
          {kind === "page" && data.files?.["index.html"] && data.files?.["shot.png"] && (
            <iframe title={t("bk_preview")} src={fileUrl(run, brief, slot, "play.html")} sandbox="allow-scripts" style={{ width: "100%", aspectRatio: "16 / 9", border: "1px solid var(--border)", borderRadius: 8, background: "#000" }} />
          )}
          {kind === "text" && data.files?.["answer.txt"] && <TextAnswer run={run} brief={brief} slot={slot} />}
          <div className="kpis">
            <div className="kpi"><b>{data.latency_ms != null ? `${fmt(data.latency_ms / 1000, lang)} s` : "—"}</b><span>{t("bk_time")}</span></div>
            <div className="kpi"><b>{data.ttft_ms != null ? `${fmt(data.ttft_ms / 1000, lang, 2)} s` : "—"}</b><span>{t("bk_ttft")}</span></div>
            <div className="kpi"><b>{fmt(data.tokens_in, lang, 0)}</b><span>{t("bk_tokens_in")}</span></div>
            <div className="kpi"><b>{fmt(data.tokens_out, lang, 0)}</b><span>{t("bk_tokens_out")}</span></div>
            <div className="kpi"><b>{fmt(data.tokens_per_s, lang)}</b><span>{t("bk_tps")}</span></div>
            <div className="kpi"><b>{money(data.cost, currency, lang)}</b><span>{t("bk_cost")}</span></div>
            {kind === "page" && <div className="kpi"><b>{auto.moves == null ? "—" : auto.moves ? t("yes") : t("no")}</b><span>{t("bk_moves")}</span></div>}
            {kind === "page" && <div className="kpi"><b>{(render.page_errors || []).length}</b><span>{t("bk_page_errors")}</span></div>}
          </div>
          {judge?.scores && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2"><h3>{t("bk_judge")}</h3><Chip>{t(data.judge_mode === "images" ? "bk_judge_images_short" : "bk_judge_text_short")}</Chip><b className="num">{fmt(data.mean, lang)}</b></div>
              <div className="grid gap-2 sm:grid-cols-2">
                {DIMENSIONS.map((d) => (
                  <div key={d}>
                    <div className="flex justify-between"><span>{t(`bk_dim_${d}`)}</span><b className="num">{fmt(judge.scores[d], lang)}</b></div>
                    <Progress done={(judge.scores[d] || 0) * 10} total={100} />
                  </div>
                ))}
              </div>
              {judge.summary && <p>{judge.summary}</p>}
              {judge.problems?.length > 0 && <div className="banner banner-warn"><b>{t("bk_problems")}</b><ul>{judge.problems.map((c, i) => <li key={i}>{c}</li>)}</ul></div>}
            </div>
          )}
          {kind === "page" && data.files?.["shot.png"] && (
            <div className="grid gap-2 sm:grid-cols-2">
              <img src={fileUrl(run, brief, slot, "shot.png")} alt={t("bk_shot_one")} style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />
              {data.files?.["shot2.png"] && <img src={fileUrl(run, brief, slot, "shot2.png")} alt={t("bk_shot_two")} style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />}
            </div>
          )}
          {kind === "page" && data.files?.["index.html"] && <a className="btn btn-sm" href={fileUrl(run, brief, slot, "index.html")}>{t("bk_download")}</a>}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ one cell of the grid
function Cell({ run, brief, row, currency, best, onPick, picked }) {
  const { t, lang } = useApp();
  return (
    <div className="panel panel-tight space-y-2" style={{ outline: picked ? "2px solid var(--accent)" : undefined }}>
      <button type="button" className="btn btn-sm" aria-expanded={picked} onClick={onPick} style={{ width: "100%", justifyContent: "flex-start", overflowWrap: "anywhere" }}><b>{row.model}</b></button>
      {row.has_shot ? (
        <img src={fileUrl(run, brief.id, row.slot, "shot.png")} alt={`${row.model}: ${brief.title}`} loading="lazy" onClick={onPick} style={{ width: "100%", aspectRatio: "16 / 9", objectFit: "cover", borderRadius: 6, border: "1px solid var(--border)", cursor: "pointer" }} />
      ) : row.has_text ? (
        <div style={{ maxHeight: 150, overflow: "hidden", position: "relative" }}><TextAnswer run={run} brief={brief.id} slot={row.slot} /></div>
      ) : (
        <div className="banner banner-warn" style={{ aspectRatio: "16 / 9", overflow: "auto" }}><Chip className={OUTCOME_CHIP[row.outcome]}>{t(`bk_out_${row.outcome || "queued"}`)}</Chip> {row.reason && <Chip>{t(`bk_reason_${row.reason}`)}</Chip>}</div>
      )}
      <div className="flex flex-wrap gap-x-3 gap-y-1 help num">
        <span>{row.latency_s != null ? `${fmt(row.latency_s, lang)} s` : "—"}</span>
        <span>{row.tokens_out != null ? `${num(row.tokens_in ?? 0, 0, lang)} → ${num(row.tokens_out, 0, lang)} ${t("bk_tok")}` : "—"}</span>
        <span>{money(row.cost, currency, lang)}</span>
        <span>{row.mean != null ? `${t("bk_score")} ${fmt(row.mean, lang)}` : ""}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {best.score === row.slot && <Chip className="chip-ok">{t("bk_best")}</Chip>}
        {best.time === row.slot && <Chip className="chip-info">{t("bk_fastest")}</Chip>}
        {best.cost === row.slot && <Chip className="chip-accent">{t("bk_cheapest")}</Chip>}
        {row.page_errors > 0 && <Chip className="chip-danger">{t("bk_page_errors")} {row.page_errors}</Chip>}
        {row.outcome && row.outcome !== "done" && row.has_shot && <Chip className={OUTCOME_CHIP[row.outcome]}>{t(`bk_out_${row.outcome}`)}</Chip>}
      </div>
    </div>
  );
}

function leaders(rows) {
  const pick = (key, low) => {
    const pool = rows.filter((r) => r[key] != null);
    if (pool.length < 2) return null;
    const sorted = [...pool].sort((a, b) => (low ? a[key] - b[key] : b[key] - a[key]));
    return sorted[0][key] === sorted[1][key] ? null : sorted[0].slot;
  };
  return { score: pick("mean", false), time: pick("latency_s", true), cost: pick("cost", true) };
}

function sortRows(rows, how) {
  const key = { score: ["mean", false], time: ["latency_s", true], cost: ["cost", true] }[how];
  if (!key) return rows;
  return [...rows].sort((a, b) => {
    if (a[key[0]] == null && b[key[0]] == null) return 0;
    if (a[key[0]] == null) return 1;
    if (b[key[0]] == null) return -1;
    return key[1] ? a[key[0]] - b[key[0]] : b[key[0]] - a[key[0]];
  });
}

// ------------------------------------------------------------------ one run
function RunDetail({ id }) {
  const { t, lang, version, changed, notify } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("briefbench_result", { run: id }), [id, version]);
  const run = data?.run;
  usePoll(reload, 2500, !!run && ACTIVE_STATES.has(run.state));
  const [pick, setPick] = useState(null);
  const [sort, setSort] = useState("order");
  const [busy, doRun] = useBusy();
  if (!data) return <div className="space-y-3"><a href="#/encargos" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_briefs")}</a><ErrorBox error={error} />{loading && !error && <Spinner />}</div>;
  const s = data.summary;
  const currency = run.currency;
  const active = ACTIVE_STATES.has(run.state);
  const cancel = () => doRun("cancel", async () => { await api.call("briefbench_cancel", { run: id }); notify(t("bk_cancelled")); changed(); reload(); });
  const copy = async () => {
    try { await navigator.clipboard.writeText(data.markdown); notify(t("hb_copied")); } catch { notify(t("hb_copy_failed"), "error"); }
  };
  const phase = run.progress.phase;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/encargos" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_briefs")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{t.msg(run.label)}</h1>
        <StateChip state={run.state} />
        <button type="button" className="btn" onClick={copy}><Icon d={ICONS.copy} size={14} />{t("hb_copy")}</button>
        {active && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("bk_cancel")}</Busy>}
      </div>
      <ErrorBox error={error} />
      {run.error && <Failure m={run.error} />}
      <div className="panel space-y-2">
        <Progress done={run.done} total={run.total} />
        <div className="flex flex-wrap gap-4 help num">
          <span>{run.done}/{run.total}</span>
          <span style={{ fontWeight: phase === "answering" ? 700 : 400 }}>{t("bk_phase_answering")} {run.progress.answered ?? 0}/{run.n}</span>
          {run.judge && <span style={{ fontWeight: phase === "judging" ? 700 : 400 }}>{t("bk_phase_judging")} {run.progress.judged ?? 0}/{run.n}</span>}
          {run.progress.current && active && <span>{t("bk_now")}: {run.progress.current}</span>}
          {run.judge && <span>{t("bk_judge")}: {run.judge.name}{run.same_model_judges ? ` (${t("bk_same_short")})` : ""}</span>}
        </div>
        {(run.progress.warnings || []).map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
      </div>
      <Section title={t("bk_summary")}>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("bk_model")}</th><th className="r">{t("bk_answers")}</th><th className="r">{t("bk_time_mean")}</th><th className="r">{t("bk_tokens_in")}</th><th className="r">{t("bk_tokens_out")}</th><th className="r">{t("bk_tps")}</th><th className="r">{t("bk_cost_total")}</th><th className="r">{t("bk_score")}</th></tr></thead>
            <tbody>
              {s.models.map((m) => (
                <tr key={m.slot}>
                  <td style={{ overflowWrap: "anywhere" }}>{m.name} {m.kind === "cli" && <Chip>{t("bk_cli")}</Chip>}
                    <div className="flex flex-wrap gap-1">
                      {s.best_score === m.name && <Chip className="chip-ok">{t("bk_best")}</Chip>}
                      {s.fastest === m.name && <Chip className="chip-info">{t("bk_fastest")}</Chip>}
                      {s.cheapest === m.name && <Chip className="chip-accent">{t("bk_cheapest")}</Chip>}
                    </div></td>
                  <td className="r num">{m.usable}/{m.briefs}</td>
                  <td className="r num">{m.latency_s_mean != null ? `${fmt(m.latency_s_mean, lang)} s` : "—"}</td>
                  <td className="r num">{fmt(m.tokens_in_total, lang, 0)}</td>
                  <td className="r num">{fmt(m.tokens_out_total, lang, 0)}</td>
                  <td className="r num">{fmt(m.tokens_per_s, lang)}</td>
                  <td className="r num">{money(m.cost_total, currency, lang)}{m.cost_total != null && m.cost_unknown > 0 ? ` (+${m.cost_unknown} ?)` : ""}</td>
                  <td className="r num">{fmt(m.mean_score, lang)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="help">{t("bk_legend")}{s.same_model_judges ? ` ${t("bk_same_warn")}` : ""}</p>
      </Section>
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="bk-sort" className="help">{t("bk_sort")}</label>
        <select id="bk-sort" className="field" style={{ width: "auto" }} value={sort} onChange={(e) => setSort(e.target.value)}>
          {SORTS.map((x) => <option key={x} value={x}>{t(`bk_sort_${x}`)}</option>)}
        </select>
      </div>
      {data.briefs.map((b) => {
        const best = leaders(b.items);
        const rows = sortRows(b.items, sort);
        const shown = pick && pick.brief === b.id ? pick : null;
        return (
          <Section key={b.id} title={b.title} count={b.items.length}>
            <div className="flex flex-wrap items-center gap-2"><Chip>{t(`bk_kind_${b.kind}`)}</Chip><a className="btn btn-sm" href={b.grid_url} target="_blank" rel="noreferrer">{t("bk_grid_image")}</a></div>
            <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}>
              {rows.map((row) => <Cell key={row.slot} run={id} brief={b} row={row} currency={currency} best={best} picked={shown?.slot === row.slot} onPick={() => setPick(shown?.slot === row.slot ? null : { brief: b.id, slot: row.slot, kind: b.kind })} />)}
            </div>
            {shown && <ItemDetail key={`${shown.brief}-${shown.slot}`} run={id} brief={shown.brief} kind={shown.kind} slot={shown.slot} currency={currency} onClose={() => setPick(null)} />}
          </Section>
        );
      })}
    </div>
  );
}

export default function Encargos({ param }) {
  const { t, version } = useApp();
  const catalog = useLoad(() => api.call("briefbench_catalog", {}), [version]);
  if (param) return <RunDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_briefs")}</h1>
      <p className="help">{t("bk_help")}</p>
      <NewRun catalog={catalog.data} />
      <History />
    </div>
  );
}
