import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, toNumber } from "../format.js";
import { ArtifactSelect, PostSteps, buildAfter, emptyAfter, useArtifacts } from "../components/cards.jsx";
import { Busy, Check, Chip, ErrorBox, Field, Icon, ICONS, PageHead, Section, Seg, Tabs, useBusy } from "../components/ui.jsx";
import { MERGE_METHODS } from "../meta.js";

const MODEL_KINDS = ["base", "merged", "ctx_variant"];

function LoraMerge({ query }) {
  const { t, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [adapter, setAdapter] = useState(query.get("adapter") || "");
  const [base, setBase] = useState("");
  const [device, setDevice] = useState("");
  const [name, setName] = useState("");
  const [after, setAfter] = useState(emptyAfter());
  const pickAdapter = async (id) => {
    setAdapter(id);
    if (!id) return;
    try {
      const a = await api.call("artifact_get", { artifact: id });
      const parent = (a.ancestors || []).find((x) => MODEL_KINDS.includes(x.kind));
      if (parent) setBase(parent.id);
    } catch { /* the user can pick the base by hand */ }
  };
  const start = () => run("go", async () => {
    const args = { base, adapter, ...(device ? { device } : {}), ...(name.trim() ? { name: name.trim() } : {}) };
    const a = buildAfter(after, { allowMerge: false });
    if (a) args.after = a;
    const r = await api.call("merge_lora_start", args);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });
  return (
    <div className="space-y-4">
      <div className="panel space-y-3">
        <p className="help">{t("lora_merge_help")}</p>
        <div className="grid gap-3 md:grid-cols-2">
          <ArtifactSelect label={t("adapter")} kinds={["adapter"]} value={adapter} onChange={pickAdapter} />
          <ArtifactSelect label={t("base_model")} kinds={MODEL_KINDS} value={base} onChange={setBase} hint={t("merge_base_hint")} />
          <Field label={t("merge_device")} hint={t("merge_device_hint")}>
            <select className="field" value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="">{t("from_settings")}</option>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </Field>
          <Field label={t("result_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        </div>
      </div>
      <Section id="lora-after" title={t("next_steps")}>
        <div className="panel"><PostSteps value={after} onChange={setAfter} showMerge={false} /></div>
      </Section>
      <Busy className="btn btn-primary" busy={busy.go} disabled={!adapter || !base} onClick={start}><Icon d={ICONS.merge} size={14} />{t("start_merge")}</Busy>
    </div>
  );
}

function ModelsMerge() {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const { data } = useArtifacts(MODEL_KINDS);
  const [models, setModels] = useState([]);       // [{ id, weight }]
  const [method, setMethod] = useState("linear");
  const [base, setBase] = useState("");
  const [tt, setT] = useState("0.5");
  const [density, setDensity] = useState("0.5");
  const [lam, setLam] = useState("1");
  const [seed, setSeed] = useState("");
  const [normalize, setNormalize] = useState(true);
  const [name, setName] = useState("");
  const [after, setAfter] = useState(emptyAfter());
  const [check, setCheck] = useState(null);
  const [checkError, setCheckError] = useState(null);
  const all = data || [];
  const byId = Object.fromEntries(all.map((a) => [a.id, a]));
  const needsBase = method === "ties" || method === "dare";

  const add = (id) => { if (id && !models.some((m) => m.id === id)) { setModels([...models, { id, weight: "1" }]); setCheck(null); } };
  const remove = (id) => { setModels(models.filter((m) => m.id !== id)); setCheck(null); };
  const setWeight = (id, weight) => setModels(models.map((m) => (m.id === id ? { ...m, weight } : m)));

  const doCheck = () => run("check", async () => {
    setCheckError(null);
    try {
      setCheck(await api.call("merge_check", { models: models.map((m) => m.id), ...(needsBase && base ? { base } : {}) }));
    } catch (e) {
      setCheck(null);
      setCheckError(e);
    }
  });
  const start = () => run("go", async () => {
    const args = { models: models.map((m) => m.id), method, ...(name.trim() ? { name: name.trim() } : {}) };
    if (method !== "slerp") args.weights = models.map((m) => toNumber(m.weight) ?? 1);
    if (method === "slerp") args.t = toNumber(tt) ?? 0.5;
    if (needsBase) args.base = base;
    if (needsBase) args.density = toNumber(density) ?? 0.5;
    if (method === "ties") args.lam = toNumber(lam) ?? 1;
    if (seed !== "") args.seed = toNumber(seed);
    if (method === "linear") args.normalize = normalize;
    const a = buildAfter(after, { allowMerge: false });
    if (a) args.after = a;
    const r = await api.call("merge_models_start", args);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });
  const valid = models.length >= 2 && (method !== "slerp" || models.length === 2) && (!needsBase || base);
  return (
    <div className="space-y-4">
      <div className="panel space-y-3">
        <p className="help">{t("models_merge_help")}</p>
        <Field label={t("models_to_merge")}>
          <select className="field" value="" onChange={(e) => add(e.target.value)}>
            <option value="">{t("add_model")}…</option>
            {all.filter((a) => !models.some((m) => m.id === a.id)).map((a) => <option key={a.id} value={a.id}>{a.name} · {t(`kind_${a.kind}`)} · {bytes(a.size, lang)}</option>)}
          </select>
        </Field>
        {models.map((m) => (
          <div key={m.id} className="flex flex-wrap items-center gap-2">
            <span className="min-w-0 flex-1 basis-52 font-semibold" style={{ overflowWrap: "anywhere" }}>{byId[m.id]?.name || m.id}</span>
            {method !== "slerp" && <label className="check">{t("weight")}<input className="field" style={{ width: 80 }} type="number" step="any" value={m.weight} onChange={(e) => setWeight(m.id, e.target.value)} /></label>}
            <button type="button" className="btn btn-sm" onClick={() => remove(m.id)} aria-label={t("remove")}><Icon d={ICONS.x} size={12} /></button>
          </div>
        ))}
        <div>
          <span className="label">{t("merge_method")}</span>
          <Seg value={method} onChange={(v) => { setMethod(v); setCheck(null); }} label={t("merge_method")} options={MERGE_METHODS.map((m) => ({ value: m, label: t(`method_${m}`) }))} />
          <p className="help mt-1">{t(`method_${method}_help`)}</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-4">
          {method === "slerp" && <Field label="t" hint={t("slerp_t_hint")}><input className="field" type="number" min="0" max="1" step="0.05" value={tt} onChange={(e) => setT(e.target.value)} /></Field>}
          {needsBase && <ArtifactSelect className="sm:col-span-2" label={t("merge_reference")} kinds={MODEL_KINDS} value={base} onChange={setBase} hint={t("merge_reference_hint")} />}
          {needsBase && <Field label={t("density")}><input className="field" type="number" min="0.01" max="1" step="0.05" value={density} onChange={(e) => setDensity(e.target.value)} /></Field>}
          {method === "ties" && <Field label="λ"><input className="field" type="number" step="0.1" value={lam} onChange={(e) => setLam(e.target.value)} /></Field>}
          {method === "dare" && <Field label={t("seed")}><input className="field" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>}
          <Field label={t("result_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        </div>
        {method === "linear" && <Check checked={normalize} onChange={setNormalize}>{t("normalize_weights")}</Check>}
        <div className="flex flex-wrap gap-2">
          <Busy className="btn" busy={busy.check} disabled={models.length < 2} onClick={doCheck}><Icon d={ICONS.check} size={14} />{t("check_compat")}</Busy>
        </div>
        <ErrorBox error={checkError} />
        {check && (
          <div className="space-y-2" aria-live="polite">
            <div className="flex flex-wrap gap-2">
              <Chip className={check.ok ? "chip-ok" : "chip-danger"}>{check.ok ? t("compatible") : t("not_compatible")}</Chip>
              <Chip>{check.tensors} {t("tensors")}</Chip>
              <Chip>{bytes(check.bytes, lang)}</Chip>
              {check.disk_needed_gb != null && <Chip>{t("disk_needed")}: {check.disk_needed_gb} GB</Chip>}
            </div>
            {(check.problems || []).map((p, i) => <div key={i} className="banner banner-danger">{typeof p === "string" ? p : t.msg(p)}</div>)}
            {(check.dtype_differences || []).length > 0 && <div className="banner banner-warn">{t("dtype_differ")}</div>}
          </div>
        )}
      </div>
      <Section id="models-after" title={t("next_steps")}>
        <div className="panel"><PostSteps value={after} onChange={setAfter} showMerge={false} /></div>
      </Section>
      <Busy className="btn btn-primary" busy={busy.go} disabled={!valid} onClick={start}><Icon d={ICONS.merge} size={14} />{t("start_merge")}</Busy>
      {!valid && models.length > 0 && <p className="help">{method === "slerp" ? t("slerp_two") : needsBase && !base ? t("need_reference") : t("need_two_models")}</p>}
    </div>
  );
}

export default function Fusionar({ query }) {
  const { t } = useApp();
  const [tab, setTab] = useState(query.get("tab") === "models" ? "models" : "lora");
  return (
    <div className="space-y-4">
      <PageHead title={t("nav_merge")} subtitle={t("merge_sub")} />
      <Tabs value={tab} onChange={setTab} options={[{ value: "lora", label: t("tab_lora") }, { value: "models", label: t("tab_models") }]} />
      {tab === "lora" ? <LoraMerge query={query} /> : <ModelsMerge />}
    </div>
  );
}
