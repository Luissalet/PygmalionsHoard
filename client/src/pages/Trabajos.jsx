import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { mb, num, seconds } from "../format.js";
import { JobRow, StateChip, VramPanel, jobTitle } from "../components/cards.jsx";
import { LineChart } from "../components/charts.jsx";
import { usePoll } from "../components/hooks.js";
import { Bar, Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, KV, PageHead, Rel, Section, useBusy, useLoad } from "../components/ui.jsx";
import { JOB_KINDS, JOB_STATE_CLASS } from "../meta.js";

const ACTIVE = ["queued", "waiting_gpu", "running"];
const STOPPED = ["failed", "cancelled", "interrupted"];
const STATES = ["queued", "waiting_gpu", "running", "done", "failed", "cancelled", "interrupted"];

function JobDetail({ id }) {
  const { t, lang, notify, confirm, changed, version } = useApp();
  const [busy, run] = useBusy();
  const { data: job, error, reload } = useLoad(() => api.call("job_get", { job: id, log_lines: 200, curve_points: 400 }), [id, version]);
  const active = job && ACTIVE.includes(job.state);
  // The chain keeps changing after this step ends (the next one is created, runs and ends): poll while any step is live, or while steps are
  // still to be created and none has stopped the chain.
  const steps = job?.pipeline || [];
  const chainLive = steps.some((s) => ACTIVE.includes(s.state)) || (steps.some((s) => s.state === "planned") && !steps.some((s) => STOPPED.includes(s.state) || s.error));
  usePoll(reload, active || chainLive ? 2500 : 0);
  const logRef = useRef(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [job?.log_tail?.length]);

  if (!job) return <div className="help">{error ? <ErrorBox error={error} /> : t("loading")}</div>;
  const p = job.progress || {};
  const timing = job.timing;
  const curve = job.curve || { train: [], eval: [] };
  const series = [
    { name: "train", className: "line-train", points: curve.train || [] },
    { name: "eval", className: "line-eval", points: curve.eval || [], dots: true },
  ].filter((s) => s.points.length);
  const cancel = async () => {
    if (!(await confirm({ title: t("cancel_job"), message: t("cancel_job_msg"), confirmLabel: t("cancel_job") }))) return;
    run("c", async () => { await api.call("job_cancel", { job: job.id }); notify(t("cancel_requested")); changed(); reload(); });
  };
  const resume = () => run("r", async () => { await api.call("job_resume", { job: job.id }); notify(t("resumed")); changed(); reload(); });
  const remove = async () => {
    if (!(await confirm({ title: t("delete_job"), message: t("delete_job_msg") }))) return;
    run("d", async () => { await api.call("job_delete", { job: job.id, confirm: true }); notify(t("deleted")); changed(); window.location.hash = "#/trabajos"; });
  };
  const label = (k) => { for (const c of [`p_${k}`, k]) if (t(c) !== c) return t(c); return k; };
  const params = Object.entries(job.params || {}).filter(([, v]) => v !== null && v !== undefined && typeof v !== "object");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/trabajos" className="btn btn-sm"><Icon d={ICONS.back} size={13} />{t("nav_jobs")}</a>
        <h2 style={{ fontSize: 18 }}>{jobTitle(t, job)}</h2>
        <Chip>{t(`job_${job.kind}`)}</Chip>
        <StateChip state={job.state} />
        <div className="ml-auto flex flex-wrap gap-2">
          {ACTIVE.includes(job.state) && <Busy className="btn btn-sm" busy={busy.c} onClick={cancel}><Icon d={ICONS.stop} size={12} />{t("cancel_job")}</Busy>}
          {job.resumable && <Busy className="btn btn-sm btn-primary" busy={busy.r} onClick={resume}><Icon d={ICONS.play} size={12} />{t("resume")}</Busy>}
          {!ACTIVE.includes(job.state) && <Busy className="btn btn-sm btn-danger" busy={busy.d} onClick={remove}><Icon d={ICONS.trash} size={12} />{t("delete")}</Busy>}
        </div>
      </div>
      {job.error && <div className="banner banner-danger">{t.msg(job.error)}{job.hint ? <span className="help block">{t("hint")}: {t.msg(job.hint)}</span> : null}</div>}
      {job.state === "waiting_gpu" && <div className="banner banner-info">{p.message ? t.msg(p.message) : t("waiting_gpu_msg")}{job.queue_position ? ` · ${t("queue_pos", { n: job.queue_position })}` : ""}</div>}

      <div className="panel space-y-3">
        <Bar pct={job.pct} label={t("progress")} tone={job.state === "failed" ? "danger" : job.state === "done" ? "ok" : ""} />
        <div className="kpis">
          <div className="kpi"><b className="num">{num(job.pct, 0, lang)} %</b><span>{t("progress")}</span></div>
          {p.step != null && p.total ? <div className="kpi"><b className="num">{p.step}/{p.total}</b><span>{t("steps")}</span></div> : null}
          {p.loss != null && <div className="kpi"><b className="num">{num(p.loss, 4, lang)}</b><span>{t("loss")}</span></div>}
          {p.eval_loss != null && <div className="kpi"><b className="num">{num(p.eval_loss, 4, lang)}</b><span>{t("eval_loss")}</span></div>}
          {timing?.source === "measured" && <div className="kpi"><b className="num">{seconds(timing.eta_s)}</b><span>{t("eta_measured")}</span></div>}
          {timing?.source === "estimate" && <div className="kpi"><b className="num">{seconds(timing.eta_s)}</b><span>{t("eta_estimated")}</span></div>}
          {timing?.source === "done" && <div className="kpi"><b className="num">{seconds(timing.elapsed_s)}</b><span>{t("elapsed")}</span></div>}
          {!timing && p.eta_s != null && job.state === "running" && <div className="kpi"><b className="num">{seconds(p.eta_s)}</b><span>{t("eta")}</span></div>}
          {timing?.tokens_per_s != null ? <div className="kpi"><b className="num">{num(timing.tokens_per_s, 0, lang)}</b><span>{timing.source === "estimate" ? t("tokens_per_s_estimated") : t("tokens_per_s")}</span></div>
            : p.tokens_per_s != null ? <div className="kpi"><b className="num">{num(p.tokens_per_s, 0, lang)}</b><span>{t("tokens_per_s")}</span></div> : null}
          {p.gpu_mem_mb ? <div className="kpi"><b className="num">{mb(p.gpu_mem_mb, lang)}</b><span>{t("gpu_memory")}</span></div> : null}
          {job.gpus?.length ? <div className="kpi"><b className="num">{job.gpus.join(", ")}</b><span>GPU</span></div> : null}
          <div className="kpi"><b className="num">{job.attempts}</b><span>{t("attempts")}</span></div>
        </div>
        {p.message && <div className="help">{t.msg(p.message)}</div>}
        {timing?.source === "estimate" && <div className="help">{t("timing_estimate_help", { time: seconds(timing.eta_s), tps: num(timing.tokens_per_s, 0, lang) })}</div>}
        {timing?.source === "measured" && <div className="help">{t("timing_measured_help", { tps: num(timing.tokens_per_s, 0, lang), est: num(timing.estimated_tokens_per_s, 0, lang) })}</div>}
        {timing?.source === "done" && timing.estimated_s != null && <div className="help">{t("timing_done_help", { time: seconds(timing.elapsed_s), est: seconds(timing.estimated_s) })}</div>}
      </div>

      {series.length > 0 && (
        <Section id="curve" title={t("loss_curve")}>
          <div className="panel space-y-1">
            <LineChart series={series} lang={lang} ariaLabel={t("loss_curve")} />
            <div className="help flex flex-wrap gap-4">
              <span><span className="dot" style={{ background: "var(--hoard-accent)" }} /> {t("train_loss")}</span>
              {curve.eval?.length > 0 && <span><span className="dot" style={{ background: "var(--hoard-info)" }} /> {t("eval_loss")}</span>}
              {curve.min_loss != null && <span>{t("min_loss")}: <b className="num">{num(curve.min_loss, 4, lang)}</b></span>}
            </div>
          </div>
        </Section>
      )}

      {job.pipeline?.length > 1 && (
        <Section id="pipeline" title={t("job_pipeline")} count={job.pipeline.length}>
          <div className="flex flex-wrap items-center gap-2">
            {job.pipeline.map((s, i) => (
              <React.Fragment key={s.id || `planned-${s.step}`}>
                {i > 0 && <Icon d={ICONS.chevron} size={13} />}
                {s.id
                  ? <a href={`#/trabajos/${s.id}`} className={`chip ${s.id === job.id ? "chip-accent" : JOB_STATE_CLASS[s.state] || ""}`} style={{ textDecoration: "none" }}>{jobTitle(t, s)} · {t(`state_${s.state}`)}</a>
                  : <span className="chip" style={{ opacity: 0.7 }}>{jobTitle(t, s)} · {t(`state_${s.state}`)}</span>}
              </React.Fragment>
            ))}
          </div>
        </Section>
      )}

      {job.vram?.total_mb ? <VramPanel plan={{ vram: job.vram }} /> : null}

      {(job.out_artifact || job.result) && (
        <Section id="result" title={t("result")}>
          <div className="panel space-y-2">
            {job.out_artifact && <a className="btn btn-sm" href={`#/linaje/${job.out_artifact}`}><Icon d={ICONS.tree} size={13} />{t("open_artifact")}</a>}
            {job.result?.summary && <KV items={Object.entries(job.result.summary).map(([k, v]) => [label(k), typeof v === "number" ? num(v, Number.isInteger(v) ? 0 : 4, lang) : String(v)])} />}
            <details><summary>JSON</summary><pre className="logbox mt-2">{JSON.stringify(job.result, null, 2)}</pre></details>
          </div>
        </Section>
      )}

      <Section id="params" title={t("parameters")}>
        <details className="panel" open={job.kind === "train"}>
          <summary>{t("show_parameters")}</summary>
          <div className="mt-2"><KV items={params.map(([k, v]) => [label(k), String(v)])} /></div>
        </details>
      </Section>

      <Section id="log" title={t("log")} count={job.log_tail?.length}>
        <div className="logbox" ref={logRef} role="log" aria-live="off" tabIndex={0}>{(job.log_tail || []).join("\n") || t("log_empty")}</div>
      </Section>
    </div>
  );
}

export default function Trabajos({ param }) {
  const { t, notify, confirm, changed, version } = useApp();
  const [busy, run] = useBusy();
  const [state, setState] = useState("");
  const [kind, setKind] = useState("");
  const { data, error, reload } = useLoad(() => api.call("jobs_list", { ...(state ? { states: [state] } : {}), ...(kind ? { kind } : {}), limit: 100 }), [state, kind, version, param]);
  const jobs = data?.jobs || [];
  usePoll(reload, jobs.some((j) => ACTIVE.includes(j.state)) ? 3000 : 0);
  if (param) return <JobDetail id={param} />;
  const cancel = async (job) => {
    if (!(await confirm({ title: t("cancel_job"), message: t("cancel_job_msg"), confirmLabel: t("cancel_job") }))) return;
    run("c", async () => { await api.call("job_cancel", { job: job.id }); notify(t("cancel_requested")); changed(); reload(); });
  };
  const resume = (job) => run("r", async () => { await api.call("job_resume", { job: job.id }); notify(t("resumed")); changed(); reload(); });
  const active = jobs.filter((j) => ACTIVE.includes(j.state));
  const history = jobs.filter((j) => !ACTIVE.includes(j.state));
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_jobs")} subtitle={t("jobs_sub")} actions={<>
        <select className="field" style={{ width: "auto" }} value={state} onChange={(e) => setState(e.target.value)} aria-label={t("state")}>
          <option value="">{t("all_states")}</option>
          {STATES.map((s) => <option key={s} value={s}>{t(`state_${s}`)}</option>)}
        </select>
        <select className="field" style={{ width: "auto" }} value={kind} onChange={(e) => setKind(e.target.value)} aria-label={t("kind")}>
          <option value="">{t("all_kinds")}</option>
          {JOB_KINDS.map((k) => <option key={k} value={k}>{t(`job_${k}`)}</option>)}
        </select>
      </>} />
      <ErrorBox error={error} />
      {data?.scheduler && (
        <div className="help">
          {Object.entries(data.scheduler.lanes || {}).map(([lane, s]) => <span key={lane} className="mr-4">{t(`lane_${lane}`)}: {s.current ? t("busy") : t("idle")} · {t("sched_queue", { n: s.queued })}</span>)}
        </div>
      )}
      <Section id="queue" title={t("queue")} count={active.length}>
        {active.length ? <div className="space-y-2">{active.map((j) => <JobRow key={j.id} job={j} onCancel={cancel} />)}</div> : <Empty>{t("no_active_jobs")}</Empty>}
      </Section>
      <Section id="history" title={t("history")} count={history.length}>
        {history.length ? <div className="space-y-2">{history.map((j) => (
          <div key={j.id} className="space-y-0">
            <JobRow job={j} onResume={resume} />
            <div className="help px-1 pt-0.5"><Rel ts={j.finished_ts || j.created_ts} /></div>
          </div>
        ))}</div> : <Empty>{t("no_history")}</Empty>}
      </Section>
    </div>
  );
}
