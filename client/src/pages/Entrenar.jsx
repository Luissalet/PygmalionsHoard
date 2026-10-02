import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { toNumber } from "../format.js";
import { ArtifactSelect, DatasetSelect, PostSteps, VramPanel, buildAfter, emptyAfter } from "../components/cards.jsx";
import { Busy, Check, ErrorBox, Field, Icon, ICONS, PageHead, Seg, useBusy, useLoad } from "../components/ui.jsx";

const NUMERIC = ["rank", "alpha", "dropout", "lr", "seq_len", "batch", "grad_accum", "epochs", "max_steps", "warmup", "weight_decay", "eval_every", "save_every", "seed"];

function Step({ n, title, children }) {
  return (
    <section className="panel space-y-3" aria-label={title}>
      <div className="flex items-center gap-2"><span className="step-num">{n}</span><h2>{title}</h2></div>
      {children}
    </section>
  );
}

export default function Entrenar({ query }) {
  const { t, notify, changed, version } = useApp();
  const [busy, run] = useBusy();
  const settings = useLoad(() => api.call("settings_get", {}), [version]);
  const defaults = settings.data?.defaults || {};
  const [base, setBase] = useState(query.get("base") || "");
  const [dataset, setDataset] = useState(query.get("dataset") || "");
  const [datasetN, setDatasetN] = useState("");
  const [method, setMethod] = useState("");
  const [values, setValues] = useState({});
  const [trainOn, setTrainOn] = useState("");
  const [targets, setTargets] = useState("");
  const [name, setName] = useState("");
  const [after, setAfter] = useState(emptyAfter());
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [force, setForce] = useState(false);
  const [advanced, setAdvanced] = useState(false);

  useEffect(() => { setPlan(null); setPlanError(null); }, [base, dataset, datasetN]);

  const overrides = () => {
    const o = {};
    if (method) o.method = method;
    for (const k of NUMERIC) { const v = toNumber(values[k]); if (v !== undefined) o[k] = v; }
    if (trainOn) o.train_on = trainOn;
    if (targets.trim()) o.target_modules = targets.split(/[\s,;]+/).filter(Boolean);
    return o;
  };
  const common = () => ({ base, dataset, ...(datasetN ? { dataset_n: Number(datasetN) } : {}), ...overrides() });

  const doPlan = () => run("plan", async () => {
    setPlanError(null);
    try {
      setPlan(await api.call("train_plan", common()));
    } catch (e) {
      setPlan(null);
      setPlanError(e);
    }
  });
  const applyPlan = () => {
    if (!plan) return;
    const p = plan.params;
    setMethod(p.method);
    setValues(Object.fromEntries(NUMERIC.map((k) => [k, p[k] != null ? String(p[k]) : ""])));
    notify(t("plan_applied"));
  };
  const start = () => run("start", async () => {
    const args = { ...common(), ...(name.trim() ? { name: name.trim() } : {}), ...(force ? { force: true } : {}) };
    const a = buildAfter(after);
    if (a) args.after = a;
    const r = await api.call("train_start", args);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });

  const num = (k, label, step = "any") => (
    <Field label={label} key={k}>
      <input className="field" type="number" step={step} value={values[k] ?? ""} placeholder={defaults[k] != null ? String(defaults[k]) : ""} onChange={(e) => setValues({ ...values, [k]: e.target.value })} />
    </Field>
  );
  const ready = base && dataset;
  const doesNotFit = plan?.pick?.fits === false;

  return (
    <div className="space-y-4">
      <PageHead title={t("nav_train")} subtitle={t("train_sub")} />
      <Step n={1} title={t("step_base")}>
        <ArtifactSelect label={t("base_model")} kinds={["base"]} value={base} onChange={setBase} hint={t("base_hint")} />
      </Step>
      <Step n={2} title={t("step_dataset")}>
        <DatasetSelect value={dataset} onChange={setDataset} n={datasetN} onN={setDatasetN} />
      </Step>
      <Step n={3} title={t("step_method")}>
        <div className="space-y-3">
          <Seg label={t("method")} value={method || defaults.method || "qlora"} onChange={setMethod} options={[{ value: "qlora", label: "QLoRA (4-bit)" }, { value: "lora", label: "LoRA (bf16)" }]} />
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {num("rank", t("p_rank"), "1")}
            {num("alpha", "alpha", "1")}
            {num("dropout", t("p_dropout"))}
            {num("lr", t("p_lr"))}
            {num("seq_len", t("p_seq_len"), "1")}
            {num("epochs", t("p_epochs"))}
          </div>
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {num("batch", t("p_batch"), "1")}
            {num("grad_accum", t("p_grad_accum"), "1")}
            {num("max_steps", t("p_max_steps"), "1")}
            {num("warmup", t("p_warmup"))}
            {num("eval_every", t("p_eval_every"), "1")}
            {num("save_every", t("p_save_every"), "1")}
          </div>
          <button type="button" className="btn-link" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>{advanced ? t("hide_advanced") : t("show_advanced")}</button>
          {advanced && (
            <div className="grid gap-3 sm:grid-cols-4">
              {num("weight_decay", t("p_weight_decay"))}
              {num("seed", t("seed"), "1")}
              <Field label={t("p_train_on")} hint={t("p_train_on_hint")}>
                <select className="field" value={trainOn} onChange={(e) => setTrainOn(e.target.value)}>
                  <option value="">{t("auto")}</option>
                  <option value="assistant">{t("train_on_assistant")}</option>
                  <option value="last">{t("train_on_last")}</option>
                </select>
              </Field>
              <Field label={t("p_targets")} hint={t("p_targets_hint")}><input className="field" value={targets} onChange={(e) => setTargets(e.target.value)} placeholder="all" /></Field>
            </div>
          )}
          <p className="help">{t("empty_uses_defaults")}</p>
          <div className="flex flex-wrap gap-2">
            <Busy className="btn btn-primary" busy={busy.plan} disabled={!ready} onClick={doPlan}><Icon d={ICONS.ruler} size={14} />{t("compute_plan")}</Busy>
            {plan && <button type="button" className="btn" onClick={applyPlan}>{t("apply_plan")}</button>}
          </div>
          <ErrorBox error={planError} />
          <VramPanel plan={plan} />
        </div>
      </Step>
      <Step n={4} title={t("step_after")}>
        <p className="help">{t("after_help")}</p>
        <PostSteps value={after} onChange={setAfter} />
      </Step>
      <Step n={5} title={t("step_start")}>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("adapter_name")} hint={t("adapter_name_hint")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        </div>
        {doesNotFit && <Check checked={force} onChange={setForce}>{t("force_queue")}</Check>}
        <div className="flex flex-wrap items-center gap-3">
          <Busy className="btn btn-primary" busy={busy.start} disabled={!ready || (doesNotFit && !force)} onClick={start}><Icon d={ICONS.play} size={14} />{t("start_training")}</Busy>
          {!ready && <span className="help">{t("pick_base_dataset")}</span>}
        </div>
      </Step>
    </div>
  );
}
