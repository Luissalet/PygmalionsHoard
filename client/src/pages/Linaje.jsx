import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { bytes, num, seconds } from "../format.js";
import { ArtifactRow, PplTable, VerdictChip } from "../components/cards.jsx";
import { Busy, Check, Chip, CopyButton, Drawer, Empty, ErrorBox, Field, Icon, ICONS, KV, KindChip, Modal, PageHead, Rel, Section, useBusy, useLoad } from "../components/ui.jsx";
import { ARTIFACT_KINDS, INTENTS } from "../meta.js";

function Graph({ graph, selected, onSelect }) {
  const { t } = useApp();
  if (!graph?.nodes?.length) return null;
  const pad = 14;
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  const colX = {};
  graph.nodes.forEach((n) => { colX[n.col] = n.x; });
  const colKinds = {};
  graph.nodes.forEach((n) => { (colKinds[n.col] = colKinds[n.col] || new Set()).add(n.kind); });
  const labels = Object.keys(colKinds).map((c) => ({ col: Number(c), text: [...colKinds[c]].map((k) => t(`kind_${k}`)).join(" / ") }));
  const curve = (a, b) => {
    const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
    const mid = (x1 + x2) / 2;
    return `M${x1 + pad},${y1 + 18} C${mid + pad},${y1 + 18} ${mid + pad},${y2 + 18} ${x2 + pad},${y2 + 18}`;
  };
  return (
    <div className="scroll-x lineage-svg">
      <svg width={graph.width + pad * 2} height={graph.height + 36} viewBox={`0 0 ${graph.width + pad * 2} ${graph.height + 36}`} role="img" aria-label={t("lineage_graph")}>
        {labels.map((c) => <text key={c.col} className="col-label" x={colX[c.col] + pad} y={14}>{c.text}</text>)}
        {graph.edges.map((e, i) => byId[e.from] && byId[e.to] ? <path key={i} className="edge" d={curve(byId[e.from], byId[e.to])} /> : null)}
        {graph.nodes.map((n) => (
          <g key={n.id} className={`node ${selected === n.id ? "sel" : ""}`} transform={`translate(${n.x + pad},${n.y + 18})`} tabIndex={0} role="button" aria-label={n.name}
            onClick={() => onSelect(n.id)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(n.id); } }}>
            <rect width={n.w} height={n.h} rx="6" />
            <text x="8" y="19">{n.name.length > 24 ? `${n.name.slice(0, 23)}…` : n.name}</text>
            <text className="sub" x="8" y="35">{n.quant || t(`kind_${n.kind}`)}{n.verdict ? ` · ${t(`verdict_${n.verdict}`)}` : ""}{n.published?.length ? " · ●" : ""}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function PublishOllama({ art, onClose, onDone }) {
  const { t, notify } = useApp();
  const [busy, run] = useBusy();
  const [f, setF] = useState({ name: "", tag: "latest", num_ctx: "", system: "" });
  const go = () => run("go", async () => {
    const r = await api.call("publish_ollama", { gguf: art.id, ...(f.name ? { name: f.name } : {}), tag: f.tag || "latest", ...(f.num_ctx ? { num_ctx: Number(f.num_ctx) } : {}), ...(f.system ? { system: f.system } : {}), wait_s: 30 });
    notify(t("published_ok", { name: r.result?.tag || r.result?.name || "" }));
    onDone();
  });
  return (
    <Modal title={t("publish_ollama")} onClose={onClose} wide>
      <p className="help">{t("publish_ollama_help")}</p>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label={t("publish_name")}><input className="field" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder={art.name} /></Field>
        <Field label={t("tag")}><input className="field" value={f.tag} onChange={(e) => setF({ ...f, tag: e.target.value })} /></Field>
        <Field label="num_ctx" hint={t("num_ctx_help")}><input className="field" type="number" min="256" value={f.num_ctx} onChange={(e) => setF({ ...f, num_ctx: e.target.value })} /></Field>
      </div>
      <Field label={t("system_prompt")}><textarea className="field" rows={3} value={f.system} onChange={(e) => setF({ ...f, system: e.target.value })} /></Field>
      <div className="flex justify-end gap-2"><button type="button" className="btn" onClick={onClose}>{t("cancel")}</button><Busy className="btn btn-primary" busy={busy.go} onClick={go}>{t("publish")}</Busy></div>
    </Modal>
  );
}

function PublishLlama({ art, onClose, onDone }) {
  const { t, notify } = useApp();
  const [busy, run] = useBusy();
  const [f, setF] = useState({ name: "", ctx: "", ngl: "99", gpu: "", extra: "" });
  const go = () => run("go", async () => {
    await api.call("publish_llama", { gguf: art.id, ...(f.name ? { name: f.name } : {}), ...(f.ctx ? { ctx: Number(f.ctx) } : {}), ngl: Number(f.ngl) || 0, ...(f.gpu !== "" ? { gpu: Number(f.gpu) } : {}), ...(f.extra.trim() ? { extra_args: f.extra.trim().split(/\s+/) } : {}) });
    notify(t("published_llama"));
    onDone();
  });
  return (
    <Modal title={t("publish_llama")} onClose={onClose} wide>
      <p className="help">{t("publish_llama_help")}</p>
      <div className="grid gap-3 sm:grid-cols-4">
        <Field label={t("publish_name")}><input className="field" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder={art.name} /></Field>
        <Field label={t("context")}><input className="field" type="number" min="512" value={f.ctx} onChange={(e) => setF({ ...f, ctx: e.target.value })} /></Field>
        <Field label="ngl"><input className="field" type="number" min="0" value={f.ngl} onChange={(e) => setF({ ...f, ngl: e.target.value })} /></Field>
        <Field label="GPU" hint={t("gpu_allowed_only")}><input className="field" type="number" min="0" value={f.gpu} onChange={(e) => setF({ ...f, gpu: e.target.value })} /></Field>
      </div>
      <Field label={t("extra_args")}><input className="field mono" value={f.extra} onChange={(e) => setF({ ...f, extra: e.target.value })} placeholder="--rope-scaling yarn" /></Field>
      <div className="flex justify-end gap-2"><button type="button" className="btn" onClick={onClose}>{t("cancel")}</button><Busy className="btn btn-primary" busy={busy.go} onClick={go}>{t("publish")}</Busy></div>
    </Modal>
  );
}

// What Galton said about a result: the verdict, how big the difference is and, for a result trained on a dataset, the verdict on the dataset's own
// held-out records and the one on a general suite apart.
function GaltonBlock({ galton, art, isGguf }) {
  const { t, lang } = useApp();
  const ds = galton.dataset;
  const reg = galton.regression;
  const stats = (g) => [
    [t("difference"), g.diff != null ? `${num(g.diff, 3, lang)}${Array.isArray(g.diff_ci) ? ` [${g.diff_ci.map((x) => num(x, 3, lang)).join(", ")}]` : ""}` : ""],
    ["p", g.p_value != null ? num(g.p_value, 4, lang) : ""], ["n", g.n],
    [t("wins_losses"), g.wins != null ? `${g.wins} / ${g.losses} / ${g.ties}` : ""],
  ];
  return (
    <div className="panel space-y-2">
      <div className="flex flex-wrap items-center gap-2"><h3>{ds ? t("verdict_dataset") : t("galton_verdict")}</h3><VerdictChip verdict={galton.verdict} /><span className="help">{t("vs")} {galton.parent_name}{galton.reference?.mode ? ` (${t(`ref_mode_${galton.reference.mode}`)}, ${galton.reference.quant})` : ""}</span></div>
      {ds && <div className="help">{t("dataset_cases", { n: ds.cases, dataset: ds.dataset, version: ds.n })}</div>}
      <KV items={[...stats(galton), [t("suites"), (galton.suites || []).join(", ")]]} />
      {(galton.warnings || []).length > 0 && <div className="help" title={galton.warnings.map((w) => t.msg(w)).join("\n")}>{t("galton_warnings", { n: galton.warnings.length })}</div>}
      {reg && (
        <div className="space-y-1 border-t pt-2" style={{ borderColor: "var(--hoard-border)" }}>
          <div className="flex flex-wrap items-center gap-2"><h3>{t("verdict_regression")}</h3><VerdictChip verdict={reg.verdict} /><span className="help">{reg.suite}</span></div>
          <KV items={stats(reg)} />
          {reg.verdict === "worse" && <div className="banner banner-warn">{t("regression_worse")}</div>}
        </div>
      )}
      {galton.verdict !== "worse" && reg?.verdict !== "worse" && !art.published?.length && isGguf && <div className="banner banner-info">{t("promote_suggested")}</div>}
    </div>
  );
}

function Detail({ id, onClose }) {
  const { t, lang, notify, confirm, changed, version } = useApp();
  const [busy, run] = useBusy();
  const { data: art, error, reload } = useLoad(() => api.call("artifact_get", { artifact: id }), [id, version]);
  const [notes, setNotes] = useState("");
  const [name, setName] = useState("");
  const [modal, setModal] = useState(null);
  const [intent, setIntent] = useState("");
  const [regression, setRegression] = useState(true);
  // what the evaluation would compare with, for the chosen intent (read-only; nothing is queued)
  const { data: evalPlan } = useLoad(async () => (art && art.kind === "gguf" ? api.call("evaluate_plan", { artifact: id, ...(intent ? { intent } : {}) }).catch(() => null) : null), [id, intent, version, art?.kind]);
  useEffect(() => { if (art) { setNotes(art.notes || ""); setName(art.name); } }, [art?.id, art?.notes, art?.name]);
  if (!art) return <Drawer title={id} onClose={onClose}>{error ? <ErrorBox error={error} /> : <div className="help">{t("loading")}</div>}</Drawer>;
  const after = () => { changed(); reload(); };
  const save = () => run("s", async () => { await api.call("artifact_update", { artifact: art.id, name, notes }); notify(t("saved")); after(); });
  const pin = () => run("p", async () => { await api.call("artifact_update", { artifact: art.id, pinned: !art.pinned }); after(); });
  const exportRecipe = () => run("x", async () => {
    const r = await api.call("artifact_recipe", { artifact: art.id });
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${art.name.replace(/[^\w.-]+/g, "_")}.recipe.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
  const evaluate = () => run("e", async () => { const r = await api.call("evaluate_start", { artifact: art.id, ...(intent ? { intent } : {}), ...(regression ? {} : { regression: false }) }); notify(t("job_started", { id: r.job?.id || "" })); changed(); if (r.job?.id) window.location.hash = `#/trabajos/${r.job.id}`; });
  const unpublish = async (p) => {
    if (!(await confirm({ title: t("unpublish"), message: t("unpublish_msg", { name: p.name }), confirmLabel: t("unpublish") }))) return;
    run("u", async () => { await api.call("unpublish", { artifact: art.id, target: p.type === "ollama" ? "ollama" : "llama", name: p.name, confirm: true }); notify(t("unpublished")); after(); });
  };
  const remove = async () => {
    const files = art.kind !== "base" && art.kind !== "ollama";
    if (!(await confirm({ title: t("delete_artifact"), message: t(files ? "delete_artifact_files_msg" : "delete_artifact_msg", { name: art.name }) }))) return;
    run("d", async () => { await api.call("artifact_delete", { artifact: art.id, delete_files: files, confirm: true }); notify(t("deleted")); changed(); onClose(); });
  };
  const training = art.metrics?.training;
  const galton = art.metrics?.galton;
  const isGguf = art.kind === "gguf";
  return (
    <Drawer title={art.name} onClose={onClose}>
      <div className="flex flex-wrap items-center gap-1.5">
        <KindChip kind={art.kind} />
        {art.quant && <Chip className="chip-accent">{art.quant}</Chip>}
        <VerdictChip verdict={art.verdict} />
        {art.exists === false && <Chip className="chip-danger">{t("file_missing")}</Chip>}
      </div>
      <KV items={[
        [t("size"), art.size ? bytes(art.size, lang) : ""], [t("created"), <Rel ts={art.created_ts} />],
        [t("path"), <span className="mono">{art.path}</span>],
        [t("dataset"), art.dataset ? `${art.dataset.dataset} v${art.dataset.n} · ${art.dataset.records} ${t("records_short")}` : ""],
        [t("hash"), art.dataset ? <span className="mono">{String(art.dataset.sha256).slice(0, 16)}…</span> : ""],
        [t("job"), art.job_id ? <a href={`#/trabajos/${art.job_id}`}>{art.job_id}</a> : ""],
      ]} />
      <div className="flex flex-wrap gap-2">
        <button type="button" className="btn btn-sm" onClick={pin}><Icon d={ICONS.pin} size={13} />{art.pinned ? t("unpin") : t("pin")}</button>
        <Busy className="btn btn-sm" busy={busy.x} onClick={exportRecipe}><Icon d={ICONS.download} size={13} />{t("export_recipe")}</Busy>
        {art.kind === "adapter" && <a className="btn btn-sm btn-primary" href={`#/fusionar?adapter=${art.id}`}><Icon d={ICONS.merge} size={13} />{t("merge_into_base")}</a>}
        {art.kind === "base" && <a className="btn btn-sm btn-primary" href={`#/entrenar?base=${art.id}`}><Icon d={ICONS.bolt} size={13} />{t("train_this")}</a>}
        {(art.kind === "merged" || art.kind === "base") && <a className="btn btn-sm" href={`#/cuantizar?model=${art.id}`}><Icon d={ICONS.compress} size={13} />{t("quantize_this")}</a>}
        {isGguf && !art.quant && <a className="btn btn-sm" href={`#/cuantizar?gguf=${art.id}`}><Icon d={ICONS.compress} size={13} />{t("quantize_this")}</a>}
      </div>

      {training && (
        <div className="panel space-y-1"><h3>{t("training")}</h3>
          <KV items={[
            [t("steps"), training.steps], [t("final_loss"), training.final_loss != null ? num(training.final_loss, 4, lang) : ""], [t("best_eval_loss"), training.best_eval_loss != null ? num(training.best_eval_loss, 4, lang) : ""],
            [t("tokens_seen"), training.tokens_seen ? num(training.tokens_seen, 0, lang) : ""], [t("duration"), training.time_s != null ? seconds(training.time_s) : ""], [t("method"), training.method],
          ]} />
        </div>
      )}
      {art.metrics?.ppl && (art.ppl_family?.length > 1
        ? <PplTable rows={art.ppl_family} currentId={art.id} />
        : <div className="panel"><KV items={[["PPL", `${num(art.metrics.ppl.value, 3, lang)} ± ${num(art.metrics.ppl.error, 3, lang)}`], [t("context"), art.metrics.ppl.ctx], [t("chunks"), art.metrics.ppl.chunks]]} /></div>)}
      {art.metrics?.gguf && <div className="panel"><KV items={[[t("architecture"), art.metrics.gguf.architecture], [t("native_context"), art.metrics.gguf.context_length], [t("layers"), art.metrics.gguf.block_count], ["file_type", art.metrics.gguf.file_type]]} /></div>}

      {galton && <GaltonBlock galton={galton} art={art} isGguf={isGguf} />}

      {isGguf && (
        <div className="panel space-y-2">
          <h3>{t("evaluate")}</h3>
          <p className="help">{t("evaluate_help")}</p>
          <div className="flex flex-wrap items-end gap-2">
            <Field label={t("intent")} className="min-w-[200px] flex-1">
              <select className="field" value={intent} onChange={(e) => setIntent(e.target.value)}>
                {INTENTS.map((i) => <option key={i} value={i}>{i ? t(`intent_${i}`) : t("intent_auto")}</option>)}
              </select>
            </Field>
            <Busy className="btn" busy={busy.e} onClick={evaluate}>{t("evaluate")}</Busy>
          </div>
          {evalPlan?.reference && (
            <div className="space-y-1" data-eval-reference={evalPlan.reference.mode}>
              <div className="help"><strong>{t("eval_reference")}:</strong> {t.msg(evalPlan.reference.message)}</div>
              {(evalPlan.reference.notes || []).map((n, i) => <div key={i} className="help">{t.msg(n)}</div>)}
              {!evalPlan.reference.ready && (evalPlan.reference.steps || []).length > 0 && <div className="help">{t("eval_ref_steps", { steps: evalPlan.reference.steps.map((k) => t(`job_${k}`)).join(", ") })}</div>}
              {(evalPlan.reference.warnings || []).map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
            </div>
          )}
          {(intent === "" || intent === "dataset") && (
            <>
              <p className="help">{t("eval_dataset_help")}</p>
              <Check checked={regression} onChange={setRegression}>{t("eval_regression")}</Check>
            </>
          )}
        </div>
      )}

      {isGguf && (
        <div className="panel space-y-2">
          <h3>{t("publish")}</h3>
          <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-sm btn-primary" onClick={() => setModal("ollama")}>Ollama</button>
            <button type="button" className="btn btn-sm" onClick={() => setModal("llama")}>llama.cpp</button>
          </div>
          {art.published?.map((p) => (
            <div key={`${p.type}${p.name}`} className="flex flex-wrap items-center gap-2">
              <Chip className="chip-ok">{p.type}</Chip><span className="mono">{p.name}</span>
              <button type="button" className="btn btn-sm ml-auto" onClick={() => unpublish(p)}>{t("unpublish")}</button>
            </div>
          ))}
        </div>
      )}

      <div className="panel space-y-2">
        <Field label={t("name")}><input className="field" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label={t("notes")}><textarea className="field" rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
        <Busy className="btn btn-sm btn-primary" busy={busy.s} disabled={name === art.name && notes === (art.notes || "")} onClick={save}>{t("save")}</Busy>
      </div>

      {art.recipe && (
        <details className="panel">
          <summary>{t("recipe")}</summary>
          <pre className="logbox mt-2">{JSON.stringify(art.recipe, null, 2)}</pre>
        </details>
      )}
      {art.ancestors?.length > 0 && (
        <div className="space-y-1.5"><h3>{t("ancestors")}</h3>{art.ancestors.map((a) => <ArtifactRow key={a.id} a={a} compact onOpen={(x) => { window.location.hash = `#/linaje/${x.id}`; }} />)}</div>
      )}
      {art.descendants?.length > 0 && (
        <div className="space-y-1.5"><h3>{t("descendants")}</h3>{art.descendants.map((a) => <ArtifactRow key={a.id} a={a} compact onOpen={(x) => { window.location.hash = `#/linaje/${x.id}`; }} />)}</div>
      )}
      <div className="flex justify-end"><Busy className="btn btn-sm btn-danger" busy={busy.d} onClick={remove}><Icon d={ICONS.trash} size={13} />{t("delete")}</Busy></div>
      {modal === "ollama" && <PublishOllama art={art} onClose={() => setModal(null)} onDone={() => { setModal(null); after(); }} />}
      {modal === "llama" && <PublishLlama art={art} onClose={() => setModal(null)} onDone={() => { setModal(null); after(); }} />}
    </Drawer>
  );
}

export default function Linaje({ param }) {
  const { t, version } = useApp();
  const [open, setOpen] = useState(param || null);
  const [kind, setKind] = useState("");
  const [text, setText] = useState("");
  const [pinned, setPinned] = useState(false);
  const graph = useLoad(() => api.call("lineage_graph", {}), [version]);
  const list = useLoad(() => api.call("artifacts_list", { ...(kind ? { kind } : {}), ...(text ? { text } : {}), ...(pinned ? { pinned: true } : {}), limit: 200 }), [kind, text, pinned, version]);
  const rows = list.data?.artifacts || [];
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_lineage")} subtitle={t("lineage_sub")} />
      <ErrorBox error={graph.error} />
      <Section id="graph" title={t("lineage_graph")}>
        {graph.data?.nodes?.length ? <Graph graph={graph.data} selected={open} onSelect={setOpen} /> : graph.loading ? <div className="help">{t("loading")}</div> : <Empty>{t("no_artifacts")}</Empty>}
      </Section>
      <Section id="table" title={t("artifacts")} count={rows.length} actions={<>
        <input className="field" style={{ width: 180 }} placeholder={t("search")} value={text} onChange={(e) => setText(e.target.value)} aria-label={t("search")} />
        <select className="field" style={{ width: "auto" }} value={kind} onChange={(e) => setKind(e.target.value)} aria-label={t("kind")}>
          <option value="">{t("all_kinds")}</option>
          {ARTIFACT_KINDS.map((k) => <option key={k} value={k}>{t(`kind_${k}`)}</option>)}
        </select>
        <Check checked={pinned} onChange={setPinned}>{t("only_pinned")}</Check>
      </>}>
        <ErrorBox error={list.error} />
        {rows.length ? <div className="space-y-2">{rows.map((a) => <ArtifactRow key={a.id} a={a} onOpen={(x) => setOpen(x.id)} />)}</div> : !list.loading && <Empty>{t("no_artifacts")}</Empty>}
      </Section>
      {open && <Detail id={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
