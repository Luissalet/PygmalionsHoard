import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, compact, num } from "../format.js";
import { ArtifactRow, VramPanel } from "../components/cards.jsx";
import { Busy, Chip, Drawer, Empty, ErrorBox, Field, Icon, ICONS, KV, Modal, PageHead, Section, useBusy, useLoad } from "../components/ui.jsx";
import { useTool } from "../components/hooks.js";

function BaseBadges({ b }) {
  const { t } = useApp();
  return (
    <div className="flex flex-wrap gap-1.5">
      <Chip className={b.trainable ? "chip-ok" : "chip-amber"} title={b.trainable ? "" : t("not_trainable_hint")}>{b.trainable ? t("trainable") : t("not_trainable")}</Chip>
      <Chip className={b.convertible ? "chip-ok" : "chip-amber"} title={t.msg(b.convert_note)}>{b.convertible ? t("convertible") : t("not_convertible")}</Chip>
      {b.multimodal && <Chip>{t("multimodal")}</Chip>}
      {b.moe && <Chip>MoE</Chip>}
    </div>
  );
}

function BaseDetail({ id, onClose }) {
  const { t, lang } = useApp();
  const { data, error } = useLoad(() => api.call("base_get", { base: id }), [id]);
  const local = data?.local;
  return (
    <Drawer title={local?.repo_id || local?.name || id} onClose={onClose}>
      <ErrorBox error={error} />
      {!data && !error && <div className="help">{t("loading")}</div>}
      {local && (
        <>
          <BaseBadges b={local} />
          {local.convert_note && <p className="help">{t.msg(local.convert_note)}</p>}
          <KV items={[
            [t("architecture"), `${local.architecture}${local.model_type ? ` (${local.model_type})` : ""}`],
            [t("parameters"), local.params ? `${num(local.params_b, 2, lang)} B` : "—"],
            [t("context"), local.context ? num(local.context, 0, lang) : "—"],
            [t("layers"), local.layers], [t("hidden"), local.hidden], [t("vocab"), local.vocab ? num(local.vocab, 0, lang) : ""],
            [t("dtype"), local.dtype], [t("size"), bytes(local.size, lang)], [t("folder"), <span className="mono">{local.path}</span>],
          ]} />
          <div className="flex flex-wrap gap-2">
            <a className="btn btn-primary" href={`#/entrenar?base=${data.artifact.id}`}><Icon d={ICONS.bolt} size={14} />{t("train_this")}</a>
            <a className="btn" href={`#/contexto?model=${data.artifact.id}`}><Icon d={ICONS.ruler} size={14} />{t("extend_context")}</a>
            <a className="btn" href={`#/cuantizar?model=${data.artifact.id}`}><Icon d={ICONS.compress} size={14} />{t("quantize_this")}</a>
          </div>
          {data.estimate && <VramPanel plan={{ vram: data.estimate }} />}
          {data.children?.length > 0 && (
            <div className="space-y-2">
              <h3>{t("derived")}</h3>
              {data.children.map((a) => <ArtifactRow key={a.id} a={a} compact onOpen={(x) => { window.location.hash = `#/linaje/${x.id}`; }} />)}
            </div>
          )}
        </>
      )}
    </Drawer>
  );
}

function DownloadModal({ info, onClose, onConfirm, busy }) {
  const { t } = useApp();
  return (
    <Modal title={t("download_model")} onClose={onClose} wide>
      <KV items={[
        [t("repository"), info.repo_id], [t("download_size"), `${info.download_gb} GB`], [t("free_disk"), `${info.free_disk_gb} GB`],
        [t("license"), info.license || "—"], [t("architecture"), info.architecture || "—"],
        [t("gated"), info.gated ? t("yes") : t("no")],
      ]} />
      {info.gated && <div className="banner banner-warn">{t("gated_hint")}</div>}
      {info.has_python_files && <div className="banner banner-warn">{t("python_files_hint")}</div>}
      <p className="help">{t("download_only_safe")}</p>
      <div className="flex justify-end gap-2">
        <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
        <Busy className="btn btn-primary" busy={busy} onClick={onConfirm}><Icon d={ICONS.download} size={14} />{t("download_n", { n: info.download_gb })}</Busy>
      </div>
    </Modal>
  );
}

function HfSearch() {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [form, setForm] = useState({ query: "", pipeline: "text-generation", sort: "downloads" });
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const search = (e) => {
    e?.preventDefault();
    if (!form.query.trim()) return;
    setError(null);
    run("s", async () => {
      try {
        const r = await api.call("hf_search", { query: form.query.trim(), pipeline: form.pipeline, sort: form.sort, limit: 20 });
        setResults(r.results || []);
      } catch (err) {
        setResults(null);
        setError(err);
      }
    });
  };
  const ask = (repo) => run(`d-${repo}`, async () => {
    const r = await api.call("base_download", { repo_id: repo });
    setInfo(r);
  });
  const confirm = () => run("go", async () => {
    const r = await api.call("base_download", { repo_id: info.repo_id, confirm: true });
    setInfo(null);
    notify(t("job_started", { id: r.job?.id || "" }));
    changed();
    if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`;
  });

  return (
    <Section id="hf" title={t("hf_search")}>
      <form className="panel space-y-3" onSubmit={search}>
        <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto_auto] sm:items-end">
          <Field label={t("search")}><input className="field" value={form.query} onChange={(e) => set("query", e.target.value)} placeholder="qwen 4b instruct" /></Field>
          <Field label={t("pipeline")}>
            <select className="field" value={form.pipeline} onChange={(e) => set("pipeline", e.target.value)}>
              <option value="text-generation">{t("pipe_text")}</option>
              <option value="image-text-to-text">{t("pipe_multi")}</option>
            </select>
          </Field>
          <Field label={t("sort_by")}>
            <select className="field" value={form.sort} onChange={(e) => set("sort", e.target.value)}>
              {["downloads", "likes", "lastModified", "trendingScore"].map((s) => <option key={s} value={s}>{t(`sort_${s}`)}</option>)}
            </select>
          </Field>
          <Busy type="submit" className="btn btn-primary" busy={busy.s}><Icon d={ICONS.search} size={14} />{t("search")}</Busy>
        </div>
        <ErrorBox error={error} />
        {results && (
          results.length ? (
            <div className="scroll-x">
              <table>
                <thead><tr><th>{t("repository")}</th><th>{t("architecture")}</th><th className="r">{t("parameters")}</th><th className="r">{t("size")}</th><th className="r">{t("downloads")}</th><th>{t("license")}</th><th /></tr></thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.repo_id}>
                      <td style={{ overflowWrap: "anywhere" }}><b>{r.repo_id}</b>{r.gated && <Chip className="chip-amber">{t("gated")}</Chip>}{!r.has_safetensors && <Chip className="chip-amber">{t("no_safetensors")}</Chip>}</td>
                      <td>{r.architecture || "—"}</td>
                      <td className="r num">{r.params ? `${num(r.params / 1e9, 1, lang)} B` : "—"}</td>
                      <td className="r num">{r.approx_bytes ? bytes(r.approx_bytes, lang) : "—"}</td>
                      <td className="r num">{compact(r.downloads, lang)}</td>
                      <td>{r.license || "—"}</td>
                      <td className="r"><Busy className="btn btn-sm" busy={busy[`d-${r.repo_id}`]} onClick={() => ask(r.repo_id)}><Icon d={ICONS.download} size={13} />{t("download")}</Busy></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="help">{t("no_results")}</div>
        )}
      </form>
      {info && <DownloadModal info={info} onClose={() => setInfo(null)} onConfirm={confirm} busy={busy.go} />}
    </Section>
  );
}

export default function Modelos({ param }) {
  const { t, lang } = useApp();
  const { data, error, loading } = useTool("bases_list", {}, []);
  const [open, setOpen] = useState(param || null);
  const bases = data?.bases || [];
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_bases")} subtitle={t("bases_sub")} />
      <Section id="local" title={t("local_bases")} count={bases.length}>
        <ErrorBox error={error} />
        {loading && !data ? <div className="help">{t("loading")}</div> : bases.length ? (
          <div className="card-grid">
            {bases.map((b) => (
              <button key={b.path} type="button" className="panel pick space-y-2" aria-pressed={open === b.artifact} onClick={() => setOpen(b.artifact || b.name)}>
                <div className="font-semibold" style={{ overflowWrap: "anywhere" }}>{b.repo_id || b.name}</div>
                <div className="help flex flex-wrap gap-x-3">
                  <span>{b.architecture}</span>
                  {b.params ? <span className="num">{num(b.params_b, 2, lang)} B</span> : null}
                  <span className="num">{bytes(b.size, lang)}</span>
                  {b.context ? <span className="num">{t("ctx")} {compact(b.context, lang)}</span> : null}
                </div>
                <BaseBadges b={b} />
              </button>
            ))}
          </div>
        ) : <Empty>{t("no_bases")}</Empty>}
        {data?.work_dir && <p className="help">{t("bases_folder")}: <span className="mono">{data.work_dir}</span></p>}
      </Section>
      <HfSearch />
      {open && <BaseDetail id={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
