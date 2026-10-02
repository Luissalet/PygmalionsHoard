import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, num, toNumber } from "../format.js";
import { ArtifactSelect, PplCell, VerdictChip, useArtifacts } from "../components/cards.jsx";
import { Busy, Check, Chip, Empty, Field, Icon, ICONS, PageHead, Rel, Section, Seg, useBusy, useLoad } from "../components/ui.jsx";
import { OUT_TYPES, QUANT_TYPES } from "../meta.js";

const HF_KINDS = ["base", "merged", "ctx_variant"];

function ResultsTable() {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const { data, loading } = useArtifacts(["gguf"]);
  const rows = data || [];
  const ppl = (a) => run(`p-${a.id}`, async () => { const r = await api.call("perplexity_start", { gguf: a.id }); notify(t("job_started", { id: r.job?.id || "" })); changed(); });
  const evaluate = (a) => run(`e-${a.id}`, async () => { const r = await api.call("evaluate_start", { artifact: a.id }); notify(t("job_started", { id: r.job?.id || "" })); changed(); });
  if (loading && !data) return <div className="help">{t("loading")}</div>;
  if (!rows.length) return <Empty>{t("no_ggufs")}</Empty>;
  return (
    <div className="scroll-x panel">
      <table>
        <thead><tr><th>{t("file")}</th><th>{t("quant")}</th><th className="r">{t("size")}</th><th className="r">PPL ± {t("error")}</th><th>{t("galton_verdict")}</th><th>{t("created")}</th><th /></tr></thead>
        <tbody>
          {rows.map((a) => (
            <tr key={a.id}>
              <td style={{ overflowWrap: "anywhere" }}><a href={`#/linaje/${a.id}`}>{a.name}</a></td>
              <td>{a.quant ? <Chip className="chip-accent">{a.quant}</Chip> : <Chip>{t("unquantized")}</Chip>}</td>
              <td className="r num">{bytes(a.size, lang)}</td>
              <td className="r num"><PplCell a={a} /></td>
              <td><VerdictChip verdict={a.verdict} /></td>
              <td><Rel ts={a.created_ts} /></td>
              <td className="r" style={{ whiteSpace: "nowrap" }}>
                <Busy className="btn btn-sm" busy={busy[`p-${a.id}`]} onClick={() => ppl(a)}>{t("measure_ppl")}</Busy>{" "}
                <Busy className="btn btn-sm" busy={busy[`e-${a.id}`]} onClick={() => evaluate(a)} title={t("evaluate_hint")}>{t("evaluate")}</Busy>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Cuantizar({ query }) {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [source, setSource] = useState(query.get("gguf") ? "gguf" : query.get("model") ? "hf" : "gguf");
  const [gguf, setGguf] = useState(query.get("gguf") || "");
  const [model, setModel] = useState(query.get("model") || "");
  const [types, setTypes] = useState(["Q4_K_M"]);
  const [imatrix, setImatrix] = useState(true);
  const [calibration, setCalibration] = useState("bundled");
  const [calibDataset, setCalibDataset] = useState("");
  const [chunks, setChunks] = useState("");
  const [perplexity, setPerplexity] = useState(true);
  const [outtype, setOutType] = useState("f16");
  const [name, setName] = useState("");
  const all = useArtifacts(["gguf", ...HF_KINDS]);
  const datasets = useLoad(() => (calibration === "dataset" ? api.call("datasets_list", {}) : Promise.resolve(null)), [calibration]);
  const chosen = (all.data || []).find((a) => a.id === (source === "gguf" ? gguf : model));
  const needsMatrix = types.some((id) => QUANT_TYPES.find((q) => q.id === id)?.imatrix);
  const useMatrix = imatrix || needsMatrix;
  const toggle = (id) => setTypes(types.includes(id) ? types.filter((x) => x !== id) : [...types, id]);
  // The source is stored at 16 bits per weight (f16/bf16); a quantized type shrinks it by bits/16.
  const estimate = (bits) => (chosen?.size ? bytes((chosen.size * bits) / 16, lang) : "—");
  const valid = types.length > 0 && (source === "gguf" ? gguf : model) && (!useMatrix || calibration !== "dataset" || calibDataset);

  const start = () => run("go", async () => {
    const args = { types, imatrix: useMatrix, perplexity, ...(name.trim() ? { name: name.trim() } : {}) };
    if (source === "gguf") args.gguf = gguf; else { args.model = model; args.outtype = outtype; }
    if (useMatrix) args.calibration = calibration === "dataset" ? { source: "dataset", dataset: calibDataset } : { source: "bundled" };
    const c = toNumber(chunks);
    if (c) args.chunks = c;
    const r = await api.call("quantize_start", args);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });

  return (
    <div className="space-y-6">
      <PageHead title={t("nav_quantize")} subtitle={t("quantize_sub")} />
      <div className="panel space-y-4">
        <Seg value={source} onChange={setSource} label={t("source")} options={[{ value: "gguf", label: t("from_gguf") }, { value: "hf", label: t("from_hf") }]} />
        {source === "gguf"
          ? <ArtifactSelect label={t("gguf_file")} kinds={["gguf"]} filter={(a) => !a.quant} value={gguf} onChange={setGguf} hint={t("gguf_source_hint")} />
          : <div className="grid gap-3 sm:grid-cols-2">
            <ArtifactSelect label={t("hf_model")} kinds={HF_KINDS} value={model} onChange={setModel} hint={t("hf_source_hint")} />
            <Field label={t("step_convert")}><select className="field" value={outtype} onChange={(e) => setOutType(e.target.value)}>{OUT_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}</select></Field>
          </div>}
        <div>
          <span className="label">{t("quant_types")}</span>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {QUANT_TYPES.map((q) => (
              <label key={q.id} className={`panel panel-tight check pick`} aria-pressed={types.includes(q.id)} style={{ display: "flex" }}>
                <input type="checkbox" checked={types.includes(q.id)} onChange={() => toggle(q.id)} />
                <span className="min-w-0 flex-1"><b>{q.id}</b>{q.imatrix ? " *" : ""}<span className="help block">{q.bits} bpw · ≈ {estimate(q.bits)}</span></span>
              </label>
            ))}
          </div>
          <span className="help mt-1 block">{t("quant_star_hint")} {t("estimate_note")}</span>
        </div>
        <div className="space-y-2">
          <Check checked={useMatrix} disabled={needsMatrix} onChange={setImatrix}>{t("step_imatrix")}</Check>
          {useMatrix && (
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label={t("calibration")}>
                <select className="field" value={calibration} onChange={(e) => setCalibration(e.target.value)}>
                  <option value="bundled">{t("calib_bundled")}</option>
                  <option value="dataset">{t("calib_dataset")}</option>
                </select>
              </Field>
              {calibration === "dataset" && (
                <Field label={t("dataset")}>
                  <select className="field" value={calibDataset} onChange={(e) => setCalibDataset(e.target.value)}>
                    <option value="">{t("pick_one")}</option>
                    {(datasets.data?.datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                </Field>
              )}
              <Field label={t("chunks")} hint={t("chunks_hint")}><input className="field" type="number" min="1" value={chunks} onChange={(e) => setChunks(e.target.value)} /></Field>
            </div>
          )}
        </div>
        <Check checked={perplexity} onChange={setPerplexity}>{t("step_ppl")}</Check>
        <div className="grid gap-3 sm:grid-cols-2"><Field label={t("result_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field></div>
        <Busy className="btn btn-primary" busy={busy.go} disabled={!valid} onClick={start}><Icon d={ICONS.compress} size={14} />{t("start_quantize")}</Busy>
      </div>
      <Section id="results" title={t("quant_results")}>
        <ResultsTable />
      </Section>
    </div>
  );
}
