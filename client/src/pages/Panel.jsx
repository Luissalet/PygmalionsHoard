import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes } from "../format.js";
import { ArtifactRow, GpuStrip, JobRow } from "../components/cards.jsx";
import { Busy, Chip, Empty, ErrorBox, Icon, ICONS, PageHead, Section, useBusy } from "../components/ui.jsx";
import { EnvReport } from "./Ajustes.jsx";

function StatusChip({ ok, label, detail, unknown }) {
  const cls = unknown ? "" : ok ? "chip-ok" : "chip-amber";
  return <Chip className={cls} title={detail}>{label}</Chip>;
}

export default function Panel() {
  const { t, lang, dash, dashError, changed, notify, confirm } = useApp();
  const [busy, run] = useBusy();
  const [report, setReport] = useState(null);

  const check = () => run("env", async () => {
    const r = await api.call("env_check", { fresh: true });
    setReport(r);
    changed();
  });
  const cancel = async (job) => {
    if (!(await confirm({ title: t("cancel_job"), message: t("cancel_job_msg"), confirmLabel: t("cancel_job") }))) return;
    run("c", async () => { await api.call("job_cancel", { job: job.id }); notify(t("cancel_requested")); changed(); });
  };
  const resume = (job) => run("r", async () => { await api.call("job_resume", { job: job.id }); notify(t("resumed")); changed(); });

  if (!dash) return <div className="help">{dashError ? <ErrorBox error={dashError} /> : t("loading")}</div>;
  const env = dash.env || {};
  const caps = env.capabilities || {};
  const checked = !!env.checked;
  const galton = dash.galton || {};
  const steps = [];
  if (!checked) steps.push({ text: t("next_check_env"), href: null, action: check });
  else if (!env.ok) steps.push({ text: t("next_fix_env"), href: "#/ajustes" });
  if (!dash.counts.bases) steps.push({ text: t("next_get_base"), href: "#/modelos" });
  if (!dash.counts.datasets) steps.push({ text: t("next_make_dataset"), href: "#/datasets" });
  if (dash.counts.bases && dash.counts.datasets && !dash.counts.artifacts) steps.push({ text: t("next_train"), href: "#/entrenar" });

  return (
    <div className="space-y-6">
      <PageHead title={t("nav_panel")} subtitle={t("panel_sub")} actions={<>
        <a className="btn btn-primary" href="#/entrenar"><Icon d={ICONS.bolt} size={15} />{t("qa_train")}</a>
        <a className="btn" href="#/cuantizar"><Icon d={ICONS.compress} size={15} />{t("qa_quantize")}</a>
        <a className="btn" href="#/contexto"><Icon d={ICONS.ruler} size={15} />{t("qa_context")}</a>
      </>} />

      {dash.offline && <div className="banner banner-info">{t("offline_banner")}</div>}

      <Section id="env" title={t("environment")} actions={<Busy className="btn btn-sm" busy={busy.env} onClick={check}><Icon d={ICONS.refresh} size={13} />{t("env_check")}</Busy>}>
        <div className="panel space-y-2">
          <div className="flex flex-wrap gap-2">
            <StatusChip unknown={!checked} ok={env.ok} label={`${t("trainer_env")}: ${!checked ? t("not_checked") : env.ok ? t("ok") : t("env_problems", { n: env.problems })}`} />
            <StatusChip unknown={!checked} ok={env.cuda} label={`CUDA: ${!checked ? "—" : env.cuda ? (env.torch || t("ok")) : t("no")}`} />
            <StatusChip unknown={!checked} ok={caps.quantize && caps.imatrix} label={`llama.cpp: ${!checked ? "—" : caps.quantize ? t("ok") : t("missing")}`} />
            <StatusChip unknown={!checked} ok={caps.convert} label={`${t("convert_scripts")}: ${!checked ? "—" : caps.convert ? t("ok") : t("missing")}`} />
            <StatusChip unknown={!checked} ok={caps.publish_ollama} label={`Ollama: ${!checked ? "—" : caps.publish_ollama ? t("ok") : t("missing")}`} />
            <StatusChip ok={galton.ok} label={`Galton: ${galton.ok ? t("reachable") : t("unreachable_short")}`} detail={t.msg(galton.detail)} />
            <StatusChip ok={dash.gpus?.hub} label={`Hub: ${dash.gpus?.hub ? t("reachable") : t("unreachable_short")}`} />
          </div>
          {!galton.ok && galton.detail ? <div className="help">Galton: {t.msg(galton.detail)}</div> : null}
          {steps.length > 0 && (
            <ul className="m-0 list-disc space-y-1 pl-5">
              {steps.map((s, i) => <li key={i}>{s.href ? <a href={s.href}>{s.text}</a> : <button type="button" className="btn-link" onClick={s.action}>{s.text}</button>}</li>)}
            </ul>
          )}
          {report && <EnvReport report={report} />}
        </div>
      </Section>

      <Section id="gpus" title="GPU" actions={<a className="btn btn-sm" href="#/ajustes">{t("gpu_settings")}</a>}>
        <GpuStrip status={dash.gpus} />
      </Section>

      <Section id="running" title={t("active_jobs")} count={dash.jobs.length} actions={<a className="btn btn-sm" href="#/trabajos">{t("all_jobs")}</a>}>
        {dash.jobs.length ? <div className="space-y-2">{dash.jobs.map((j) => <JobRow key={j.id} job={j} onCancel={cancel} />)}</div> : <Empty>{t("no_active_jobs")}</Empty>}
      </Section>

      {dash.failed.length > 0 && (
        <Section id="failed" title={t("failed_jobs")} count={dash.failed.length}>
          <div className="space-y-2">{dash.failed.map((j) => <JobRow key={j.id} job={j} onResume={resume} />)}</div>
        </Section>
      )}

      <Section id="latest" title={t("latest_artifacts")} count={dash.counts.artifacts} actions={<a className="btn btn-sm" href="#/linaje">{t("nav_lineage")}</a>}>
        {dash.artifacts.length ? <div className="space-y-2">{dash.artifacts.slice(0, 8).map((a) => <ArtifactRow key={a.id} a={a} onOpen={(x) => { window.location.hash = `#/linaje/${x.id}`; }} />)}</div> : <Empty>{t("no_artifacts")}</Empty>}
      </Section>

      <Section id="counts" title={t("studio")}>
        <div className="kpis">
          <div className="kpi"><b className="num">{dash.counts.bases}</b><span>{t("nav_bases")}</span></div>
          <div className="kpi"><b className="num">{dash.counts.datasets}</b><span>{t("nav_datasets")}</span></div>
          <div className="kpi"><b className="num">{dash.counts.artifacts}</b><span>{t("nav_lineage")}</span></div>
          <div className="kpi"><b className="num">{dash.counts.jobs}</b><span>{t("nav_jobs")}</span></div>
          <div className="kpi"><b className="num">{bytes(dash.storage?.total, lang)}</b><span>{t("disk_used")}</span></div>
        </div>
      </Section>
    </div>
  );
}
