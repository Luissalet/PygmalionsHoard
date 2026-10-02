import React, { useEffect, useMemo } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, mb, num, seconds, signed } from "../format.js";
import { JOB_STATE_CLASS, OUT_TYPES, QUANT_TYPES, VERDICT_CLASS } from "../meta.js";
import { Sparkline } from "./charts.jsx";
import { Bar, Chip, Field, Icon, ICONS, KindChip, Rel, useLoad } from "./ui.jsx";

export function VerdictChip({ verdict }) {
  const { t } = useApp();
  if (!verdict) return null;
  return <Chip className={VERDICT_CLASS[verdict] || ""} title={t("galton_verdict")}>{t(`verdict_${verdict}`)}</Chip>;
}

export function StateChip({ state }) {
  const { t } = useApp();
  return <Chip className={JOB_STATE_CLASS[state] || ""}>{t(`state_${state}`)}</Chip>;
}

// The title of a job: the app's own wording of what it does (a coded message), else the kind of job.
export function jobTitle(t, job) {
  const plain = job.title && typeof job.title === "object" ? job.title.text : job.title;
  if (!plain || plain === job.kind) return t(`job_${job.kind}`);
  const text = t.msg(job.title);
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// A perplexity with its error, and underneath how it moved against the nearest measured ancestor and against the base model's own file.
export function PplCell({ a }) {
  const { t, lang } = useApp();
  if (!a.ppl) return <>—</>;
  const move = (cmp, label) => cmp && <div className="help num" title={`${label}: ${cmp.name}`}>{label} {signed(cmp.delta, 3, lang)}{cmp.pct != null ? ` (${signed(cmp.pct, 1, lang)} %)` : ""}</div>;
  return (
    <div>
      <div>{num(a.ppl.value, 3, lang)} ± {num(a.ppl.error, 3, lang)}</div>
      {move(a.ppl_cmp?.parent, t("ppl_vs_parent"))}
      {move(a.ppl_cmp?.base, t("ppl_vs_base"))}
    </div>
  );
}

// The f16 file, each quantization and the base model's own files side by side, measured the same way.
export function PplTable({ rows, currentId }) {
  const { t, lang } = useApp();
  if (!rows?.length) return null;
  const move = (cmp) => (cmp ? `${signed(cmp.delta, 3, lang)}${cmp.pct != null ? ` (${signed(cmp.pct, 1, lang)} %)` : ""}` : "—");
  return (
    <div className="panel space-y-1">
      <h3>{t("ppl_table")}</h3>
      <div className="help">{t("ppl_table_help")}</div>
      <div className="scroll-x">
        <table>
          <thead><tr><th>{t("file")}</th><th>{t("quant")}</th><th className="r">PPL ± {t("error")}</th><th className="r">{t("ppl_vs_parent")}</th><th className="r">{t("ppl_vs_base")}</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} style={r.id === currentId ? { fontWeight: 700 } : undefined}>
                <td style={{ overflowWrap: "anywhere" }}><a href={`#/linaje/${r.id}`}>{r.name}</a>{!r.dataset_version && <> <Chip>{t("ppl_base_file")}</Chip></>}</td>
                <td>{r.quant ? <Chip className="chip-accent">{r.quant}</Chip> : <Chip>{t("unquantized")}</Chip>}</td>
                <td className="r num">{num(r.ppl.value, 3, lang)} ± {num(r.ppl.error, 3, lang)}</td>
                <td className="r num">{move(r.ppl_cmp?.parent)}</td>
                <td className="r num">{move(r.ppl_cmp?.base)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// A job in a list: what it is, where it stands, how long it will take.
export function JobRow({ job, onCancel, onResume }) {
  const { t, lang } = useApp();
  const p = job.progress || {};
  const active = ["running", "waiting_gpu"].includes(job.state);
  const title = jobTitle(t, job);
  const timing = job.timing;
  return (
    <div className="panel panel-tight space-y-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <a href={`#/trabajos/${job.id}`} className="min-w-0 flex-1 basis-48 font-semibold" style={{ overflowWrap: "anywhere" }}>{title}</a>
        <Chip>{t(`job_${job.kind}`)}</Chip>
        <StateChip state={job.state} />
        {job.state === "queued" && job.queue_position ? <Chip>{t("queue_pos", { n: job.queue_position })}</Chip> : null}
        {job.gpus?.length ? <Chip>GPU {job.gpus.join(", ")}</Chip> : null}
        {job.curve?.length > 1 && <Sparkline points={job.curve} ariaLabel={t("loss")} />}
        {onCancel && active && <button type="button" className="btn btn-sm" onClick={() => onCancel(job)}><Icon d={ICONS.stop} size={12} />{t("cancel_job")}</button>}
        {onResume && ["interrupted", "failed", "cancelled"].includes(job.state) && <button type="button" className="btn btn-sm" onClick={() => onResume(job)}><Icon d={ICONS.play} size={12} />{t("resume")}</button>}
      </div>
      {job.state === "running" && (
        <div className="space-y-1">
          <Bar pct={job.pct} thin label={t("progress")} />
          <div className="help flex flex-wrap gap-x-4">
            <span className="num">{num(job.pct, 0, lang)} %</span>
            {p.step != null && p.total ? <span className="num">{p.step}/{p.total}</span> : null}
            {p.loss != null && <span>{t("loss")} <span className="num">{num(p.loss, 3, lang)}</span></span>}
            {timing?.source === "measured" ? <span>{t("eta_measured")} {seconds(timing.eta_s)} · <span className="num">{num(timing.tokens_per_s, 0, lang)}</span> {t("tokens_per_s")}</span>
              : timing?.source === "estimate" ? <span>{t("eta_estimated")} {seconds(timing.eta_s)}</span>
              : p.eta_s != null ? <span>{t("eta")} {seconds(p.eta_s)}</span> : null}
            {p.gpu_mem_mb ? <span>{mb(p.gpu_mem_mb, lang)}</span> : null}
            {p.message ? <span>{t.msg(p.message)}</span> : null}
          </div>
        </div>
      )}
      {job.state === "waiting_gpu" && <div className="help">{p.message ? t.msg(p.message) : t("waiting_gpu_msg")}{timing?.source === "estimate" ? ` · ${t("eta_estimated")} ${seconds(timing.eta_s)}` : ""}</div>}
      {job.state === "queued" && timing?.source === "estimate" && <div className="help">{t("eta_estimated")} {seconds(timing.eta_s)}</div>}
      {job.error && ["failed", "interrupted"].includes(job.state) && <div className="help" style={{ color: "var(--hoard-danger)", overflowWrap: "anywhere" }}>{t.msg(job.error)}{job.hint ? ` — ${t.msg(job.hint)}` : ""}</div>}
    </div>
  );
}

// An artifact in a list.
export function ArtifactRow({ a, onOpen, compact = false }) {
  const { t, lang } = useApp();
  const open = onOpen ? () => onOpen(a) : undefined;
  return (
    <div className="panel panel-tight flex flex-wrap items-center gap-x-3 gap-y-1">
      <div className="min-w-0 flex-1 basis-56">
        {open ? <button type="button" className="btn-link text-left" style={{ overflowWrap: "anywhere" }} onClick={open}>{a.name}</button> : <span className="font-semibold">{a.name}</span>}
        <div className="help flex flex-wrap gap-x-3">
          <Rel ts={a.created_ts} />
          {a.size ? <span className="num">{bytes(a.size, lang)}</span> : null}
          {a.exists === false && <span style={{ color: "var(--hoard-danger)" }}>{t("file_missing")}</span>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <KindChip kind={a.kind} />
        {a.quant && <Chip className="chip-accent">{a.quant}</Chip>}
        {!compact && a.ppl && <Chip title={t("ppl")}>PPL {num(a.ppl.value, 2, lang)} ± {num(a.ppl.error, 2, lang)}</Chip>}
        {!compact && a.final_loss != null && <Chip>{t("loss")} {num(a.final_loss, 3, lang)}</Chip>}
        <VerdictChip verdict={a.verdict} />
        {a.published?.length ? <Chip className="chip-ok">{t("published")}</Chip> : null}
        {a.pinned && <Chip className="chip-accent"><Icon d={ICONS.pin} size={11} />{t("pinned")}</Chip>}
      </div>
    </div>
  );
}

// Artifacts of some kinds, loaded once.
export function useArtifacts(kinds, extra = {}) {
  const { version } = useApp();
  const list = Array.isArray(kinds) ? kinds : kinds ? [kinds] : [null];
  return useLoad(async () => {
    const parts = await Promise.all(list.map((kind) => api.call("artifacts_list", { ...(kind ? { kind } : {}), limit: 200, ...extra })));
    const seen = new Set();
    const items = [];
    for (const part of parts) for (const a of part.artifacts || []) if (!seen.has(a.id)) { seen.add(a.id); items.push(a); }
    return items.sort((x, y) => (y.created_ts || 0) - (x.created_ts || 0));
  }, [list.join(","), version, JSON.stringify(extra)]);
}

export function ArtifactSelect({ label, hint, value, onChange, kinds, empty, filter, className }) {
  const { t } = useApp();
  const { data, loading } = useArtifacts(kinds);
  const items = (data || []).filter((a) => (filter ? filter(a) : true));
  return (
    <Field label={label} hint={hint} className={className}>
      <select className="field" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={loading && !data}>
        <option value="">{empty ?? (items.length ? t("pick_one") : t("none_available"))}</option>
        {items.map((a) => <option key={a.id} value={a.id}>{a.name} · {t(`kind_${a.kind}`)}{a.quant ? ` ${a.quant}` : ""}</option>)}
      </select>
    </Field>
  );
}

// Dataset + version pickers.
export function DatasetSelect({ value, onChange, n, onN, label, hint }) {
  const { t } = useApp();
  const { data } = useLoad(() => api.call("datasets_list", {}), []);
  const list = data?.datasets || [];
  const current = list.find((d) => d.id === value);
  const versions = useLoad(() => (value ? api.call("dataset_get", { dataset: value }) : Promise.resolve(null)), [value]);
  const items = versions.data?.dataset?.version_list || [];
  return (
    <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
      <Field label={label || t("dataset")} hint={hint}>
        <select className="field" value={value || ""} onChange={(e) => { onChange(e.target.value); onN?.(""); }}>
          <option value="">{list.length ? t("pick_one") : t("none_available")}</option>
          {list.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.latest ? `${d.latest.usable} ${t("records_short")}` : ""}</option>)}
        </select>
      </Field>
      {onN && (
        <Field label={t("version")}>
          <select className="field" value={n ?? ""} onChange={(e) => onN(e.target.value)} disabled={!current}>
            <option value="">{t("latest")}</option>
            {items.map((v) => <option key={v.id} value={v.n}>v{v.n} · {v.usable} {t("records_short")}</option>)}
          </select>
        </Field>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ GPUs
export function GpuStrip({ status }) {
  const { t, lang } = useApp();
  if (!status) return null;
  const { gpus = [], allowed = [], reserved = [], queue = [] } = status;
  if (!gpus.length) return <div className="panel help">{t("no_gpu")}</div>;
  return (
    <div className="space-y-2">
      <div className="gpu-strip">
        {gpus.map((g) => {
          const isAllowed = allowed.includes(g.index);
          const isReserved = reserved.includes(g.index) && !isAllowed;
          const used = g.total_mb ? ((g.total_mb - (g.free_mb ?? g.total_mb)) / g.total_mb) * 100 : 0;
          return (
            <div key={g.index} className={`gpu ${isAllowed ? "gpu-allowed" : ""} ${isReserved || !isAllowed ? "gpu-reserved" : ""}`}>
              <div className="flex items-center gap-2">
                <span className="font-semibold">GPU {g.index}</span>
                <span className="chip" style={{ marginLeft: "auto" }}>{isAllowed ? t("gpu_allowed") : reserved.includes(g.index) ? t("gpu_reserved") : t("gpu_off")}</span>
              </div>
              <div className="help trunc" title={g.name}>{g.name}</div>
              <Bar pct={used} tone={used > 90 ? "danger" : used > 70 ? "warn" : ""} thin label={`GPU ${g.index}`} />
              <div className="help num">{mb(g.free_mb, lang)} {t("free")} / {mb(g.total_mb, lang)}</div>
              {g.leases?.map((l, i) => <div key={i} className="help">{l.owner}{l.purpose ? ` · ${l.purpose}` : ""}{l.vram_mb ? ` · ${mb(l.vram_mb, lang)}` : ""}</div>)}
            </div>
          );
        })}
      </div>
      {queue.length > 0 && <div className="help">{t("hub_queue", { n: queue.length })}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ memory estimate
export function VramPanel({ plan }) {
  const { t, lang } = useApp();
  if (!plan?.vram) return null;
  const v = plan.vram;
  const pick = plan.pick || {};
  return (
    <div className="panel space-y-2" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h3>{t("vram_estimate")}</h3>
        <Chip className="chip-accent">{mb(v.total_mb, lang)}</Chip>
        {pick.fits === true && <Chip className="chip-ok">{pick.gpus?.length > 1 ? t("fits_split", { gpus: pick.gpus.join(" + ") }) : t("fits_on", { gpu: pick.gpus?.[0] })}{pick.waits ? ` · ${t("will_wait")}` : ""}</Chip>}
        {pick.fits === false && <Chip className="chip-danger">{t("does_not_fit")}</Chip>}
        {pick.fits === null && <Chip>{t("no_inventory")}</Chip>}
      </div>
      <div className="formula">{t.msg(v.formula)}</div>
      {plan.fits?.length > 0 && (
        <div className="scroll-x">
          <table>
            <thead><tr><th>GPU</th><th className="r">{t("free")}</th><th className="r">{t("total")}</th><th>{t("fits_now")}</th></tr></thead>
            <tbody>
              {plan.fits.map((f) => (
                <tr key={f.gpu}>
                  <td>{f.gpu} · {f.name}</td>
                  <td className="r num">{mb(f.free_mb, lang)}</td>
                  <td className="r num">{mb(f.total_mb, lang)}</td>
                  <td>{f.fits_free ? <Chip className="chip-ok">{t("yes")}</Chip> : f.fits_total ? <Chip className="chip-amber">{t("when_free")}</Chip> : <Chip className="chip-danger">{t("no")}</Chip>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {pick.reason ? <div className="help">{t.msg(pick.reason)}</div> : null}
      {plan.time && (
        <div className="help space-y-1">
          <div>
            {t("time_estimate")}: <b className="num">{seconds(plan.time.seconds)}</b> · {plan.steps?.total ?? "?"} {t("steps")} · <span className="num">{num(plan.time.tokens_per_s, 0, lang)}</span> {t("tokens_per_s")} · {t.msg(plan.time.note)}
          </div>
          {plan.time.formula ? <details><summary>{t("est_formula")}</summary><div className="formula mt-1">{t.msg(plan.time.formula)}</div></details> : null}
        </div>
      )}
      {plan.warnings?.map((w, i) => <div key={i} className="banner banner-warn"><Icon d={ICONS.alert} size={14} /> {t.msg(w)}</div>)}
    </div>
  );
}

// ------------------------------------------------------------------ steps after a stage
export const emptyAfter = (overrides = {}) => ({
  merge: false, convert: "", quantize: [], imatrix: true, calibration: "bundled", calibDataset: "", perplexity: false,
  publish: false, target: "ollama", pubName: "", tag: "latest", evaluate: false, intent: "", regression: true, ...overrides,
});

// The `after` object the tools expect; empty parts are left out.
export function buildAfter(s, { allowMerge = true } = {}) {
  const out = {};
  if (allowMerge && (s.merge || s.convert || s.quantize.length || s.publish || s.evaluate || s.perplexity)) out.merge = true;
  if (s.convert) out.convert = s.convert;
  if (s.quantize.length) {
    out.quantize = s.quantize;
    out.imatrix = !!s.imatrix;
    if (s.imatrix && s.calibration === "dataset" && s.calibDataset) out.calibration = { source: "dataset", dataset: s.calibDataset };
    else if (s.imatrix) out.calibration = { source: "bundled" };
  }
  if (s.perplexity) out.perplexity = true;
  if (s.publish) out.publish = { target: s.target, ...(s.pubName ? { name: s.pubName } : {}), ...(s.tag && s.tag !== "latest" ? { tag: s.tag } : {}) };
  if (s.evaluate) out.evaluate = { ...(s.intent ? { intent: s.intent } : {}), ...(s.regression === false ? { regression: false } : {}) };
  return Object.keys(out).length ? out : undefined;
}

export function PostSteps({ value, onChange, showMerge = true, showConvert = true }) {
  const { t } = useApp();
  const set = (patch) => onChange({ ...value, ...patch });
  const needsImatrix = value.quantize.some((q) => QUANT_TYPES.find((x) => x.id === q)?.imatrix);
  const { data: ds } = useLoad(() => (value.calibration === "dataset" ? api.call("datasets_list", {}) : Promise.resolve(null)), [value.calibration]);
  const toggleQuant = (id) => set({ quantize: value.quantize.includes(id) ? value.quantize.filter((q) => q !== id) : [...value.quantize, id] });
  useEffect(() => {
    if (needsImatrix && !value.imatrix) onChange({ ...value, imatrix: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsImatrix]);
  return (
    <div className="space-y-3">
      {showMerge && (
        <label className="check"><input type="checkbox" checked={value.merge || value.convert !== "" || value.quantize.length > 0 || value.publish || value.evaluate || value.perplexity} disabled={value.convert !== "" || value.quantize.length > 0 || value.publish || value.evaluate || value.perplexity} onChange={(e) => set({ merge: e.target.checked })} />{t("step_merge")}</label>
      )}
      {showConvert && (
        <Field label={t("step_convert")} hint={t("step_convert_hint")}>
          <select className="field" value={value.convert} onChange={(e) => set({ convert: e.target.value })}>
            <option value="">{t("auto")}</option>
            {OUT_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </Field>
      )}
      <div>
        <span className="label">{t("step_quantize")}</span>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          {QUANT_TYPES.map((q) => <label key={q.id} className="check" title={`${q.bits} bpw`}><input type="checkbox" checked={value.quantize.includes(q.id)} onChange={() => toggleQuant(q.id)} />{q.id}{q.imatrix ? " *" : ""}</label>)}
        </div>
        <span className="help mt-1 block">{t("quant_star_hint")}</span>
      </div>
      {value.quantize.length > 0 && (
        <div className="space-y-2 rounded-md border p-2.5" style={{ borderColor: "var(--hoard-border)" }}>
          <label className="check"><input type="checkbox" checked={value.imatrix} disabled={needsImatrix} onChange={(e) => set({ imatrix: e.target.checked })} />{t("step_imatrix")}</label>
          {value.imatrix && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label={t("calibration")}>
                <select className="field" value={value.calibration} onChange={(e) => set({ calibration: e.target.value })}>
                  <option value="bundled">{t("calib_bundled")}</option>
                  <option value="dataset">{t("calib_dataset")}</option>
                </select>
              </Field>
              {value.calibration === "dataset" && (
                <Field label={t("dataset")}>
                  <select className="field" value={value.calibDataset} onChange={(e) => set({ calibDataset: e.target.value })}>
                    <option value="">{t("pick_one")}</option>
                    {(ds?.datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                </Field>
              )}
            </div>
          )}
        </div>
      )}
      <label className="check"><input type="checkbox" checked={value.perplexity} onChange={(e) => set({ perplexity: e.target.checked })} />{t("step_ppl")}</label>
      <div className="space-y-2">
        <label className="check"><input type="checkbox" checked={value.publish} onChange={(e) => set({ publish: e.target.checked })} />{t("step_publish")}</label>
        {value.publish && (
          <div className="grid gap-3 sm:grid-cols-3 rounded-md border p-2.5" style={{ borderColor: "var(--hoard-border)" }}>
            <Field label={t("target")}>
              <select className="field" value={value.target} onChange={(e) => set({ target: e.target.value })}>
                <option value="ollama">Ollama</option>
                <option value="llama">llama.cpp</option>
                <option value="both">{t("both")}</option>
              </select>
            </Field>
            <Field label={t("publish_name")}><input className="field" value={value.pubName} onChange={(e) => set({ pubName: e.target.value })} placeholder={t("same_as_model")} /></Field>
            <Field label={t("tag")}><input className="field" value={value.tag} onChange={(e) => set({ tag: e.target.value })} /></Field>
          </div>
        )}
      </div>
      <div className="space-y-2">
        <label className="check"><input type="checkbox" checked={value.evaluate} onChange={(e) => set({ evaluate: e.target.checked })} />{t("step_evaluate")}</label>
        {value.evaluate && (
          <div className="space-y-2">
            <Field label={t("intent")} hint={t("intent_hint")}>
              <select className="field" value={value.intent} onChange={(e) => set({ intent: e.target.value })}>
                <option value="">{t("intent_auto")}</option>
                {["dataset", "style", "code", "context", "general", "smoke", "quant"].map((i) => <option key={i} value={i}>{t(`intent_${i}`)}</option>)}
              </select>
            </Field>
            {(value.intent === "" || value.intent === "dataset") && (
              <label className="check"><input type="checkbox" checked={value.regression !== false} onChange={(e) => set({ regression: e.target.checked })} />{t("eval_regression")}</label>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="kpi" title={hint}>
      <b className="num">{value}</b>
      <span>{label}</span>
    </div>
  );
}

// A short line saying what a started pipeline will do, with a link to follow it.
export function startedMessage(t, answer) {
  const job = answer?.job;
  return job ? t("job_started", { id: job.id }) : t("saved");
}

export function useJobsPoll(jobs) {
  return useMemo(() => (jobs || []).some((j) => ["queued", "waiting_gpu", "running"].includes(j.state)), [jobs]);
}
