import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, compact, num } from "../format.js";
import { ArtifactSelect, PostSteps, VerdictChip, buildAfter, emptyAfter, useArtifacts } from "../components/cards.jsx";
import { Bar, Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, PageHead, Section, Seg, useBusy, useLoad } from "../components/ui.jsx";
import { CTX_LENGTHS } from "../meta.js";

const HF_KINDS = ["base", "merged"];

function FitTable() {
  const { t, lang } = useApp();
  const [gguf, setGguf] = useState("");
  const [gpu, setGpu] = useState("");
  const [busy, run] = useBusy();
  const [fit, setFit] = useState(null);
  const [error, setError] = useState(null);
  const calc = () => run("fit", async () => {
    setError(null);
    try {
      setFit(await api.call("ctx_fit", { gguf, ...(gpu !== "" ? { gpu: Number(gpu) } : {}), contexts: CTX_LENGTHS }));
    } catch (e) {
      setFit(null);
      setError(e);
    }
  });
  const s = fit?.summary;
  return (
    <div className="panel space-y-3">
      <p className="help">{t("fit_help")}</p>
      <div className="grid gap-3 sm:grid-cols-[2fr_1fr_auto] sm:items-end">
        <ArtifactSelect label={t("gguf_file")} kinds={["gguf"]} value={gguf} onChange={(v) => { setGguf(v); setFit(null); }} />
        <Field label="GPU"><input className="field" type="number" min="0" value={gpu} onChange={(e) => setGpu(e.target.value)} placeholder={t("any_allowed")} /></Field>
        <Busy className="btn btn-primary" busy={busy.fit} disabled={!gguf} onClick={calc}>{t("compute_fit")}</Busy>
      </div>
      <ErrorBox error={error} />
      {fit && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <Chip>{s.architecture}</Chip>
            <Chip>{t("native_context")}: {compact(s.context_length, lang)}</Chip>
            <Chip>{s.block_count} {t("layers")}</Chip>
            <Chip>{t("kv_heads")} {s.head_count_kv}/{s.head_count}</Chip>
            {s.rope_scaling_type && <Chip className="chip-accent">{s.rope_scaling_type} ×{s.rope_scaling_factor}</Chip>}
            {fit.free_mb != null && <Chip>{t("free")}: {num(fit.free_mb, 0, lang)} MB</Chip>}
          </div>
          <table>
            <thead><tr><th>{t("context")}</th><th className="r">KV</th><th className="r">{t("total_needed")}</th><th>{t("fits")}</th></tr></thead>
            <tbody>
              {fit.rows.map((r) => (
                <tr key={r.context}>
                  <td className="num">{compact(r.context, lang)}</td>
                  <td className="r num">{num(r.kv_mb, 0, lang)} MB</td>
                  <td className="r num">{num(r.total_mb, 0, lang)} MB</td>
                  <td>{r.fits === true ? <Chip className="chip-ok">{t("yes")}</Chip> : r.fits === false ? <Chip className="chip-danger">{t("no")}</Chip> : <Chip>—</Chip>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="help">{t.msg(fit.note)}</p>
        </div>
      )}
    </div>
  );
}

// Needle results: for each evaluated GGUF that carries them, mean score per context length, parent against variant.
function NeedleResults() {
  const { t, lang, version } = useApp();
  const { data: list } = useArtifacts(["gguf"]);
  const evaluated = (list || []).filter((a) => a.verdict).slice(0, 8);
  const { data, loading } = useLoad(async () => {
    const full = await Promise.all(evaluated.map((a) => api.call("artifact_get", { artifact: a.id })));
    return full.filter((a) => a.metrics?.galton?.needle);
  }, [evaluated.map((a) => a.id).join(","), version]);
  if (loading && !data) return <div className="help">{t("loading")}</div>;
  if (!data?.length) return <Empty>{t("no_needle")}</Empty>;
  return (
    <div className="space-y-3">
      {data.map((a) => {
        const g = a.metrics.galton;
        const ids = g.contestants || {};
        const lengths = Object.keys(g.needle).sort((x, y) => parseInt(x, 10) - parseInt(y, 10));
        return (
          <div key={a.id} className="panel space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <a href={`#/linaje/${a.id}`} className="font-semibold">{a.name}</a>
              <span className="help">{t("vs")} {g.parent_name}</span>
              <VerdictChip verdict={g.verdict} />
            </div>
            <table>
              <thead><tr><th>{t("length")}</th><th>{g.parent_name}</th><th>{a.name}</th></tr></thead>
              <tbody>
                {lengths.map((len) => {
                  const row = g.needle[len];
                  const p = row[ids.parent];
                  const c = row[ids.child];
                  const cell = (v) => (v == null ? "—" : <span className="flex items-center gap-2"><span style={{ width: 90 }}><Bar pct={v * 100} thin tone={v >= 0.8 ? "ok" : v >= 0.5 ? "warn" : "danger"} label={len} /></span><span className="num">{num(v * 100, 0, lang)} %</span></span>);
                  return <tr key={len}><td className="num">{len}</td><td>{cell(p)}</td><td>{cell(c)}</td></tr>;
                })}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}

export default function Contexto({ query }) {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [model, setModel] = useState(query.get("model") || "");
  const [mode, setMode] = useState("factor");
  const [factor, setFactor] = useState(4);
  const [target, setTarget] = useState(131072);
  const [name, setName] = useState("");
  const [after, setAfter] = useState(emptyAfter({ quantize: ["Q4_K_M"], evaluate: true, intent: "context" }));
  const base = useLoad(() => (model ? api.call("base_get", { base: model }).catch(() => null) : Promise.resolve(null)), [model]);
  const native = base.data?.local?.context;
  const result = native ? (mode === "factor" ? native * factor : target) : null;
  const start = () => run("go", async () => {
    const args = { model, ...(mode === "factor" ? { factor } : { target_length: target }), ...(name.trim() ? { name: name.trim() } : {}) };
    const a = buildAfter(after, { allowMerge: false });
    if (a) args.after = a;
    const r = await api.call("ctx_extend_start", args);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_context")} subtitle={t("context_sub")} />
      <div className="panel space-y-4">
        <ArtifactSelect label={t("hf_model")} kinds={HF_KINDS} value={model} onChange={setModel} hint={t("ctx_model_hint")} />
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <span className="label">{t("extend_by")}</span>
            <Seg value={mode} onChange={setMode} label={t("extend_by")} options={[{ value: "factor", label: t("by_factor") }, { value: "target", label: t("by_target") }]} />
          </div>
          {mode === "factor"
            ? <div><span className="label">{t("yarn_factor")}</span><Seg value={factor} onChange={setFactor} label={t("yarn_factor")} options={[2, 4, 8].map((f) => ({ value: f, label: `×${f}` }))} /></div>
            : <Field label={t("target_length")}><select className="field" value={target} onChange={(e) => setTarget(Number(e.target.value))}>{CTX_LENGTHS.map((c) => <option key={c} value={c}>{compact(c, lang)} ({c})</option>)}</select></Field>}
          {native ? <div className="help">{t("native_context")}: <b className="num">{compact(native, lang)}</b> → <b className="num">{compact(result, lang)}</b></div> : null}
        </div>
        <p className="help">{t("yarn_note")}</p>
        <div className="grid gap-3 sm:grid-cols-2"><Field label={t("result_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field></div>
        <details open>
          <summary>{t("next_steps")}</summary>
          <div className="mt-3"><PostSteps value={after} onChange={setAfter} showMerge={false} /></div>
        </details>
        <Busy className="btn btn-primary" busy={busy.go} disabled={!model} onClick={start}><Icon d={ICONS.ruler} size={14} />{t("start_ctx")}</Busy>
      </div>
      <Section id="fit" title={t("fit_table")}><FitTable /></Section>
      <Section id="needle" title={t("needle_results")}><NeedleResults /></Section>
    </div>
  );
}
