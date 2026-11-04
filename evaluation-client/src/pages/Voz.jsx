import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { num, pct } from "../format.js";

const rate = (v, lang) => (v == null ? "—" : pct(v, 1, lang));
const secs = (v, lang, d = 2) => (v == null ? "—" : num(v, d, lang));

function Summary({ run }) {
  const { t, lang } = useApp();
  const rows = Object.entries(run.summary || {});
  if (!rows.length) return <p className="help">{run.state}</p>;
  return (
    <div className="panel" style={{ overflowX: "auto" }}>
      <table className="tbl" data-speech-summary>
        <thead><tr>
          <th>{t("sb_col_engine")}</th><th>WER</th><th>CER</th><th>{t("sb_col_sdi")}</th><th>{t("sb_col_files")}</th>
          <th>{t("sb_col_load")}</th><th>{t("sb_col_first")}</th><th>RTF</th><th>{t("sb_col_fail")}</th>
        </tr></thead>
        <tbody>
          {rows.map(([engine, s]) => (
            <tr key={engine}>
              <td><b className="mono">{engine}</b>{s.error ? <span className="help block" style={{ color: "var(--danger-ink)" }}>{s.error}</span> : null}</td>
              <td>{rate(s.wer, lang)}</td><td>{rate(s.cer, lang)}</td>
              <td>{s.substitutions ?? 0} / {s.deletions ?? 0} / {s.insertions ?? 0}</td>
              <td>{s.files_done}/{run.files}</td><td>{secs(s.load_s, lang)}</td><td>{secs(s.first_text_s_median, lang)}</td>
              <td>{secs(s.rtf, lang, 3)}</td><td>{s.failures ? <Chip className="chip-danger">{s.failures}</Chip> : 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunDetail({ id }) {
  const { t, lang } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("speechbench_result", { run: id, items: true }), [id]);
  const [busy, run] = useBusy();
  const active = data && ["queued", "running"].includes(data.run.state);
  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(reload, 2000);
    return () => clearInterval(timer);
  }, [active, reload]);
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Spinner />;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ margin: 0 }}>{data.run.name} <span className="help mono">{data.run.id}</span></h2>
        <Chip className={data.run.state === "done" ? "chip-ok" : data.run.state === "failed" ? "chip-danger" : "chip-amber"}>{data.run.state}</Chip>
        {active && <Busy className="btn" busy={busy.cancel} onClick={() => run("cancel", async () => { await api.call("speechbench_cancel", { run: id }); reload(); })}>{t("sb_cancel")}</Busy>}
      </div>
      {data.run.error ? <div className="banner banner-danger">{data.run.error}</div> : null}
      <Summary run={data.run} />
      <Section title={t("sb_files")} count={data.items?.length || 0}>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table className="tbl" data-speech-items>
            <thead><tr><th>{t("sb_col_engine")}</th><th>Id</th><th>{t("sb_reference")}</th><th>{t("sb_hypothesis")}</th><th>WER</th><th>RTF</th></tr></thead>
            <tbody>
              {(data.items || []).map((i) => (
                <tr key={`${i.engine}-${i.id}`}>
                  <td className="mono">{i.engine}</td><td className="mono">{i.id}</td>
                  <td style={{ maxWidth: 360 }}>{i.reference}</td>
                  <td style={{ maxWidth: 360 }}>{i.error ? <span style={{ color: "var(--danger-ink)" }}>{i.error}</span> : i.text}</td>
                  <td>{rate(i.wer, lang)}</td><td>{secs(i.rtf, lang, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function EngineForm({ engines, onSaved }) {
  const { t, notify } = useApp();
  const [name, setName] = useState("");
  const [cmd, setCmd] = useState("");
  const [python, setPython] = useState(engines.whisper.configured || "");
  const [busy, run] = useBusy();
  const save = () => run("save", async () => {
    await api.call("speechbench_engines", { name, command: cmd.split("\n").map((s) => s.trim()).filter(Boolean) });
    setName(""); setCmd(""); notify(t("sb_saved")); onSaved();
  });
  const savePython = () => run("python", async () => { await api.call("speechbench_engines", { whisper_python: python }); notify(t("sb_saved")); onSaved(); });
  return (
    <Section title={t("sb_add_engine")}>
      <div className="panel space-y-2">
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("sb_engine_name")}><input className="field" value={name} onChange={(e) => setName(e.target.value.toLowerCase())} placeholder="r2t2-cpu" style={{ width: 160 }} /></Field>
          <Field label={t("sb_engine_cmd")} className="grow"><textarea className="field mono" rows={3} value={cmd} onChange={(e) => setCmd(e.target.value)} placeholder={"C:\\r2t2\\venv\\Scripts\\python.exe\nrun_r2t2.py\n{audio}"} /></Field>
          <Busy className="btn" busy={busy.save} disabled={!name || !cmd.includes("{audio}")} onClick={save}>{t("sb_save_engine")}</Busy>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("sb_whisper_python")} hint={engines.whisper.python || t("sb_no_whisper")} className="grow">
            <input className="field mono" value={python} onChange={(e) => setPython(e.target.value)} placeholder={"…\\venv\\Scripts\\python.exe"} />
          </Field>
          <Busy className="btn" busy={busy.python} onClick={savePython}>{t("sb_save_engine")}</Busy>
        </div>
      </div>
    </Section>
  );
}

export default function Voz() {
  const { t, lang, notify } = useApp();
  const engines = useLoad(() => api.call("speechbench_engines", {}), []);
  const runs = useLoad(() => api.call("speechbench_list", { limit: 30 }), []);
  const [folder, setFolder] = useState("");
  const [corpus, setCorpus] = useState(null);
  const [picked, setPicked] = useState(["faster-whisper:small"]);
  const [language, setLanguage] = useState("es");
  const [accents, setAccents] = useState(true);
  const [selected, setSelected] = useState(null);
  const [busy, run] = useBusy();
  const toggle = (id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const check = () => run("check", async () => setCorpus(await api.call("speechbench_corpus", { folder })));
  const start = () => run("start", async () => {
    const card = await api.call("speechbench_start", { folder, engines: picked, language: language || null, accents });
    notify(t("sb_started")); setSelected(card.id); runs.reload();
  });
  const available = engines.data?.available || [];
  return (
    <div className="space-y-5" data-speech-page>
      <div><h1>{t("sb_title")}</h1><p className="help">{t("sb_help")}</p></div>
      {engines.error && <ErrorBox error={engines.error} />}
      <div className="panel space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("sb_corpus")} hint={t("sb_corpus_hint")} className="grow">
            <input className="field mono" value={folder} onChange={(e) => { setFolder(e.target.value); setCorpus(null); }} placeholder={"D:\\voz\\corpus-es"} data-speech-folder />
          </Field>
          <Busy className="btn" busy={busy.check} disabled={!folder} onClick={check}>{t("sb_check")}</Busy>
        </div>
        {corpus && <p className="help" data-speech-corpus>{t("sb_found", { n: corpus.files, s: num(corpus.audio_s, 1, lang), p: corpus.problems.length })}</p>}
        <Field label={t("sb_engines")}>
          <div className="flex flex-wrap gap-3">
            {available.map((id) => <label key={id} className="inline-flex items-center gap-1 text-[13px]"><input type="checkbox" checked={picked.includes(id)} onChange={() => toggle(id)} /><span className="mono">{id}</span></label>)}
          </div>
        </Field>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("sb_language")}><select className="field" value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="">{t("sb_language_auto")}</option>{["es", "en", "fr", "de", "it", "pt", "ca"].map((l) => <option key={l} value={l}>{l}</option>)}
          </select></Field>
          <label className="inline-flex items-center gap-2 text-[13px]"><input type="checkbox" checked={!accents} onChange={(e) => setAccents(!e.target.checked)} />{t("sb_fold_accents")}</label>
          <Busy className="btn btn-primary" busy={busy.start} disabled={!folder || !picked.length} onClick={start} data-speech-start><Icon d={ICONS.run} size={14} />{t("sb_start")}</Busy>
        </div>
      </div>
      {engines.data && <EngineForm engines={engines.data} onSaved={engines.reload} />}
      <Section title={t("sb_runs")} count={runs.data?.count}>
        {!runs.data?.runs?.length ? <Empty>{t("sb_none")}</Empty> : (
          <div className="panel"><div className="flex flex-wrap gap-2">
            {runs.data.runs.map((r) => <button key={r.id} type="button" className={`btn btn-sm ${selected === r.id ? "btn-primary" : ""}`} onClick={() => setSelected(r.id)}>{r.name} · {r.state} · <Rel ts={r.created} /></button>)}
          </div></div>
        )}
      </Section>
      {selected && <RunDetail id={selected} />}
    </div>
  );
}
