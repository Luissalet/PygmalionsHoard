import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { mb } from "../format.js";
import { Busy, Chip, CopyButton, ErrorBox, Field, Icon, ICONS, PageHead, Section, Switch, useBusy, useLoad } from "../components/ui.jsx";

// What the trainer environment lacks and the command that fixes each gap.
export function EnvReport({ report }) {
  const { t } = useApp();
  if (!report) return null;
  const trainer = report.trainer || {};
  const libs = trainer.libs || {};
  const fixes = report.fixes || {};
  const problems = report.problems || [];
  return (
    <div className="space-y-3 rounded-md border p-3" style={{ borderColor: "var(--hoard-border)" }}>
      <div className="flex flex-wrap items-center gap-2">
        <Chip className={report.ok ? "chip-ok" : "chip-amber"}>{report.ok ? t("env_all_ok") : t("env_problems", { n: problems.length })}</Chip>
        {trainer.python && <Chip>Python {trainer.python}</Chip>}
        {trainer.torch?.version && <Chip>torch {trainer.torch.version}</Chip>}
        {trainer.disk && <Chip>{t("free_disk")}: {trainer.disk.free_gb} GB</Chip>}
      </div>
      {trainer.torch?.devices?.length > 0 && (
        <div className="help">{trainer.torch.devices.map((d) => `${d.index}: ${d.name || "?"} (${d.total_mb ? mb(d.total_mb) : "?"})`).join(" · ")}</div>
      )}
      {Object.keys(libs).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(libs).map(([name, version]) => <Chip key={name} className={version ? "" : "chip-amber"}>{name} {version || t("missing")}</Chip>)}
        </div>
      )}
      {problems.length > 0 && (
        <ul className="m-0 space-y-2 pl-0" style={{ listStyle: "none" }}>
          {problems.map((item, i) => (
            <li key={i} className="space-y-1">
              <div><b>{t.msg(item.what)}</b>: {t.msg(item.problem)}{item.optional ? ` (${t("optional")})` : ""}</div>
              {item.fix && <div className="flex flex-wrap items-center gap-2"><code className="mono">{t.msg(item.fix)}</code><CopyButton text={t.msg(item.fix)} /></div>}
            </li>
          ))}
        </ul>
      )}
      <div className="help">
        {report.llama_binaries && Object.entries(report.llama_binaries).map(([k, v]) => <div key={k}>{k}: {v || t("missing")}</div>)}
        {report.convert_scripts && Object.entries(report.convert_scripts).map(([k, v]) => <div key={k}>{k}: {v || t("missing")}</div>)}
        <div>ollama: {report.ollama || t("missing")}</div>
      </div>
      {!report.ok && Object.keys(fixes).length > 0 && (
        <details>
          <summary>{t("all_fixes")}</summary>
          <div className="mt-2 space-y-1.5">
            {Object.entries(fixes).map(([k, cmd]) => (
              <div key={k} className="flex flex-wrap items-center gap-2"><span className="help" style={{ minWidth: 110 }}>{k}</span><code className="mono">{t.msg(cmd)}</code><CopyButton text={t.msg(cmd)} /></div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

const GROUP_ORDER = ["paths", "defaults", "teacher", "galton", "publish", "jobs"];
const HINTED = new Set(["gpus.allowed", "gpus.reserved"]);

function SettingInput({ name, spec, value, onChange }) {
  const { t, lang } = useApp();
  const id = `set-${name}`;
  const key = `set_${name.replace(/\./g, "_")}`;
  const label = t(key) === key ? name : t(key);
  const hintKey = `sethint_${name.replace(/\./g, "_")}`;
  const hint = t(hintKey) !== hintKey ? t(hintKey) : lang === "en" ? spec.doc : "";
  if (spec.kind === "bool") {
    return (
      <div className="flex items-center gap-3">
        <Switch checked={value === true || value === "1" || value === "true"} onChange={(v) => onChange(v ? "1" : "0")} label={label} />
        <div><div className="font-semibold">{label}</div>{hint && <div className="help">{hint}</div>}</div>
      </div>
    );
  }
  return (
    <Field label={label} hint={hint}>
      {spec.kind === "choice" ? (
        <select id={id} className="field" value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
          {spec.choices.map((c) => <option key={c} value={c}>{c || t("auto")}</option>)}
        </select>
      ) : (
        <input id={id} className={`field ${spec.kind === "path" ? "mono" : ""}`} type={spec.kind === "int" || spec.kind === "float" ? "number" : "text"} step={spec.kind === "float" ? "any" : undefined}
          value={value ?? ""} onChange={(e) => onChange(e.target.value)} />
      )}
    </Field>
  );
}

function SettingsGroup({ group, data, reload }) {
  const { t, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const keys = (data.groups[group] || []).filter((k) => !HINTED.has(k) && k !== "ui.language" && k !== "scheduler.paused");
  const [draft, setDraft] = useState({});
  useEffect(() => setDraft({}), [data]);
  if (!keys.length) return null;
  const dirty = Object.keys(draft).length > 0;
  const save = () => run("save", async () => {
    await api.call("settings_set", { values: draft });
    notify(t("saved"));
    changed();
    reload();
  });
  return (
    <Section id={`g-${group}`} title={t(`group_${group}`)} actions={<Busy className="btn btn-sm btn-primary" busy={busy.save} disabled={!dirty} onClick={save}>{t("save")}</Busy>}>
      <div className="panel">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {keys.map((k) => <SettingInput key={k} name={k} spec={data.specs[k]} value={draft[k] ?? data.values[k]} onChange={(v) => setDraft((d) => ({ ...d, [k]: v }))} />)}
        </div>
      </div>
    </Section>
  );
}

function GpuSettings({ data, reload }) {
  const { t, notify, confirm, changed, dash } = useApp();
  const [busy, run] = useBusy();
  const allowed = String(data.values["gpus.allowed"] || "").split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const reserved = String(data.values["gpus.reserved"] || "").split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const seen = new Map();
  (dash?.gpus?.gpus || []).forEach((g) => seen.set(g.index, g.name));
  [...allowed, ...reserved].forEach((i) => { if (!seen.has(i)) seen.set(i, ""); });
  const indices = [...seen.keys()].sort((a, b) => a - b);
  const toggle = async (index, on) => {
    const next = on ? [...new Set([...allowed, index])].sort((a, b) => a - b) : allowed.filter((i) => i !== index);
    if (!next.length) { notify(t("one_gpu_at_least"), "error"); return; }
    const touchesReserved = on && reserved.includes(index);
    if (touchesReserved && !(await confirm({ title: t("reserved_gpu_title"), message: t("reserved_gpu_msg", { gpu: index }), confirmLabel: t("allow_gpu"), danger: true }))) return;
    run("g", async () => {
      await api.call("settings_set", { values: { "gpus.allowed": next }, confirm_reserved: touchesReserved });
      notify(t("saved"));
      changed();
      reload();
    });
  };
  return (
    <Section id="g-gpus" title={t("group_gpus")}>
      <div className="panel space-y-2">
        <p className="help">{t("gpus_help")}</p>
        <div className="flex flex-wrap gap-4">
          {indices.map((i) => (
            <label key={i} className="check" title={seen.get(i)}>
              <input type="checkbox" checked={allowed.includes(i)} disabled={busy.g} onChange={(e) => toggle(i, e.target.checked)} />
              GPU {i}{seen.get(i) ? ` · ${seen.get(i)}` : ""}
              {reserved.includes(i) && <Chip className="chip-amber">{t("gpu_reserved")}</Chip>}
            </label>
          ))}
        </div>
      </div>
    </Section>
  );
}

function HfToken({ data, reload }) {
  const { t, notify, confirm, changed } = useApp();
  const [busy, run] = useBusy();
  const [value, setValue] = useState("");
  const info = data.secrets?.["hf.token"] || {};
  const save = () => run("s", async () => {
    await api.call("secret_set", { value: value.trim() });
    setValue("");
    notify(t("secrets_saved"));
    changed();
    reload();
  });
  const remove = async () => {
    if (!(await confirm({ title: t("remove_token"), message: t("remove_token_msg"), confirmLabel: t("remove") }))) return;
    run("r", async () => { await api.call("secret_set", { value: "" }); notify(t("secret_removed")); changed(); reload(); });
  };
  return (
    <Section id="g-hf" title={t("hf_token")}>
      <div className="panel space-y-2">
        <p className="help">{t("hf_token_help")}</p>
        <div className="flex flex-wrap items-end gap-2">
          <Field label={t("hf_token")} className="min-w-[260px] flex-1">
            <input className="field" type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} placeholder={info.configured ? "••••••••" : "hf_…"} />
          </Field>
          <Busy className="btn btn-primary" busy={busy.s} disabled={!value.trim()} onClick={save}>{t("save")}</Busy>
          {info.configured && <Busy className="btn btn-danger" busy={busy.r} onClick={remove}>{t("remove")}</Busy>}
        </div>
        <div className="help">{info.configured ? t("token_configured", { source: info.source || "" }) : t("token_not_configured")}</div>
      </div>
    </Section>
  );
}

export default function Ajustes() {
  const { t, version, dash, changed, notify } = useApp();
  const settings = useLoad(() => api.call("settings_get", {}), [version]);
  const [busy, run] = useBusy();
  const [report, setReport] = useState(null);
  const data = settings.data;

  const check = () => run("env", async () => { setReport(await api.call("env_check", { fresh: true })); changed(); });
  useEffect(() => { api.call("env_check", {}).then((r) => { if (r && r.checked_ts) setReport(r); }).catch(() => {}); }, []);
  const togglePause = (on) => run("p", async () => { await api.call("settings_set", { values: { "scheduler.paused": on } }); notify(on ? t("sched_paused") : t("resumed")); changed(); settings.reload(); });

  if (!data) return <div className="help">{settings.error ? <ErrorBox error={settings.error} /> : t("loading")}</div>;
  const galton = dash?.galton || {};
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_settings")} subtitle={t("settings_sub")} />
      <ErrorBox error={settings.error} />

      <Section id="env" title={t("environment")} actions={<Busy className="btn btn-sm" busy={busy.env} onClick={check}><Icon d={ICONS.refresh} size={13} />{t("env_check")}</Busy>}>
        {report ? <EnvReport report={report} /> : <div className="panel help">{t("env_not_checked")}</div>}
      </Section>

      <GpuSettings data={data} reload={settings.reload} />

      <Section id="sched" title={t("queue")}>
        <div className="panel flex items-center gap-3">
          <Switch checked={data.values["scheduler.paused"] === "1" || data.values["scheduler.paused"] === "true"} onChange={togglePause} label={t("pause_queue")} />
          <div><div className="font-semibold">{t("pause_queue")}</div><div className="help">{t("pause_queue_help")}</div></div>
        </div>
      </Section>

      {GROUP_ORDER.map((g) => <SettingsGroup key={g} group={g} data={data} reload={settings.reload} />)}

      <Section id="galton-status" title={t("galton_link")}>
        <div className="panel flex flex-wrap items-center gap-2">
          <Chip className={galton.ok ? "chip-ok" : "chip-amber"}>{galton.ok ? t("reachable") : t("unreachable_short")}</Chip>
          {galton.via && <Chip>{galton.via}</Chip>}
          {galton.detail && <span className="help">{t.msg(galton.detail)}</span>}
          <button type="button" className="btn btn-sm ml-auto" onClick={changed}><Icon d={ICONS.refresh} size={13} />{t("test_connection")}</button>
        </div>
      </Section>

      <HfToken data={data} reload={settings.reload} />
    </div>
  );
}
