import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, Section, Spinner, useBusy, useLoad } from "./ui.jsx";
import { num } from "../format.js";

// ------------------------------------------------------------------ before / after chart
const BASE_COLOR = "var(--muted)";
const OTHER_COLOR = "var(--accent)";

function Bar({ value, max, color, label }) {
  const width = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-2" style={{ fontSize: 12 }}>
      <span style={{ width: 120, flex: "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={label}>{label}</span>
      <div style={{ flex: 1, background: "var(--line)", borderRadius: 3, height: 10 }}><div style={{ width: `${width}%`, height: 10, borderRadius: 3, background: color }} /></div>
      <span className="num" style={{ width: 52, textAlign: "right", flex: "none" }}>{num(value, 2)}</span>
    </div>
  );
}

export function BeforeAfter({ q, delta, base, other }) {
  const { t, lang } = useApp();
  if (!delta) return null;
  const isScore = q.type === "score";
  const title = <div className="flex flex-wrap items-center gap-2"><b>{q.id}</b><span className="help">{q.instructions}</span></div>;
  const flips = <Chip>{t("logit_flips", { n: delta.flips })}</Chip>;
  if (isScore) {
    const d = delta.expected;
    return (
      <div className="panel space-y-1.5" data-testid={`logit-chart-${q.id}`}>
        {title}
        <div className="help">{t("logit_chart_score")}</div>
        <Bar value={d.base} max={q.max} color={BASE_COLOR} label={base} />
        <Bar value={d.other} max={q.max} color={OTHER_COLOR} label={other} />
        <div className="flex flex-wrap items-center gap-2"><Chip className={d.delta > 0 ? "chip-ok" : d.delta < 0 ? "chip-danger" : ""}>Δ {d.delta > 0 ? "+" : ""}{num(d.delta, 2, lang)}</Chip>{flips}</div>
      </div>
    );
  }
  return (
    <div className="panel space-y-1.5" data-testid={`logit-chart-${q.id}`}>
      {title}
      <div className="help">{t("logit_chart_choice")}</div>
      {q.labels.map((label, i) => {
        const d = delta.probs[label];
        return (
          <div key={label} className="space-y-0.5">
            <div style={{ fontSize: 12 }}><b>{label}</b> {q.options[i]} <span className="help num">Δ {d.delta > 0 ? "+" : ""}{num(d.delta, 2, lang)}</span></div>
            <Bar value={d.base} max={1} color={BASE_COLOR} label={base} />
            <Bar value={d.other} max={1} color={OTHER_COLOR} label={other} />
          </div>
        );
      })}
      <div>{flips}</div>
    </div>
  );
}

// ------------------------------------------------------------------ result
function AggregateTable({ rubric, sets }) {
  const { t, lang } = useApp();
  return (
    <div className="panel scroll-x" style={{ padding: 0 }}>
      <table className="tbl">
        <thead><tr><th>{t("logit_question")}</th>{sets.map((s) => <th key={s.label}>{s.label}</th>)}</tr></thead>
        <tbody>
          {rubric.map((q) => (
            <tr key={q.id}>
              <td><b>{q.id}</b></td>
              {sets.map((s) => {
                const a = s.aggregate[q.id];
                if (!a) return <td key={s.label}>—</td>;
                return <td key={s.label} className="num">{q.type === "score" ? num(a.mean_expected, 2, lang) : q.labels.map((l) => `${l} ${num(a.mean_probs[l] * 100, 0, lang)} %`).join(" · ")}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ItemsTable({ rubric, sets, contexts }) {
  const { t, lang } = useApp();
  const cell = (q, a) => (!a ? "—" : q.type === "score" ? `${a.chosen} (${num(a.expected, 2, lang)})` : `${a.chosen} ${num(a.confidence * 100, 0, lang)} %`);
  return (
    <div className="panel scroll-x" style={{ padding: 0 }}>
      <table className="tbl">
        <thead><tr><th>{t("logit_item")}</th><th>{t("logit_question")}</th>{sets.map((s) => <th key={s.label}>{s.label}</th>)}</tr></thead>
        <tbody>
          {sets[0].items.flatMap((it) => rubric.map((q, qi) => (
            <tr key={`${it.id}-${q.id}`}>
              {qi === 0 && <td rowSpan={rubric.length} style={{ overflowWrap: "anywhere" }}><b>{it.id}</b><div className="help">{contexts.find((c) => c.id === it.id)?.preview}</div></td>}
              <td>{q.id}</td>
              {sets.map((s) => <td key={s.label} className="num">{cell(q, s.items.find((x) => x.id === it.id)?.answers[q.id])}</td>)}
            </tr>
          )))}
        </tbody>
      </table>
    </div>
  );
}

export function LogitResult({ data }) {
  const { t } = useApp();
  const [showItems, setShowItems] = useState(false);
  const sets = data.sets || [];
  return (
    <div className="space-y-3" data-testid="logit-result">
      {data.warnings?.map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
      {data.compare?.length > 0 && (
        <Section title={t("logit_chart_title")}>
          {data.compare.map((c) => (
            <div key={c.label} className="space-y-2">
              <div className="help">{c.label} {t("logit_against")} {c.against}</div>
              <div className="card-grid">
                {data.rubric.map((q) => <BeforeAfter key={q.id} q={q} delta={c.deltas[q.id]} base={c.against} other={c.label} />)}
              </div>
            </div>
          ))}
        </Section>
      )}
      <Section title={t("logit_totals")}><AggregateTable rubric={data.rubric} sets={sets} /></Section>
      {sets[0]?.items && (
        <Section title={t("logit_per_item")} actions={<button type="button" className="btn btn-sm" onClick={() => setShowItems((v) => !v)}>{showItems ? t("logit_hide") : t("logit_show")}</button>}>
          {showItems && <ItemsTable rubric={data.rubric} sets={sets} contexts={data.contexts || []} />}
        </Section>
      )}
    </div>
  );
}

export function LogitRunDetail({ id }) {
  const { t } = useApp();
  const { data, error, loading } = useLoad(() => api.call("logit_judge_result", { run: id, detail: true }), [id]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/ejecutar?judge=logit" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("logit_title")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{data?.run?.label || id}</h1>
        {data?.run && <Chip className={data.run.state === "done" ? "chip-ok" : "chip-danger"}>{data.run.state}</Chip>}
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data?.run?.error && <div className="banner banner-danger">{data.run.error}</div>}
      {data?.sets && <LogitResult data={data} />}
    </div>
  );
}

// ------------------------------------------------------------------ the form
const blankQuestion = (n) => ({ id: `q${n}`, type: "choice", instructions: "", options: "", min: 0, max: 9 });
const EXAMPLE = {
  rubric: [
    { id: "tono", type: "choice", instructions: "¿Qué tono tiene el texto?", options: "formal\ncoloquial\ngroseros", min: 0, max: 9 },
    { id: "calidad", type: "score", instructions: "¿Qué calidad tiene la redacción?", options: "", min: 0, max: 9 },
  ],
  items: "Estimado señor: lamentamos comunicarle que su solicitud no ha sido aprobada.\n---\nque pasa tio, no me ha llegado nada, a ver si te mueves",
};
const emptyTarget = () => ({ kind: "server", model: "", url: "", base_path: "", adapter_path: "" });

function RubricEditor({ rubric, setRubric }) {
  const { t } = useApp();
  const update = (i, patch) => setRubric((r) => r.map((q, k) => (k === i ? { ...q, ...patch } : q)));
  return (
    <div className="space-y-2">
      {rubric.map((q, i) => (
        <div key={i} className="panel space-y-2" data-testid="logit-question">
          <div className="grid gap-2 md:grid-cols-4">
            <Field label={t("logit_q_id")}><input className="field" value={q.id} onChange={(e) => update(i, { id: e.target.value })} /></Field>
            <Field label={t("logit_q_type")}>
              <select className="field" value={q.type} onChange={(e) => update(i, { type: e.target.value })}>
                <option value="choice">{t("type_choice")}</option>
                <option value="score">{t("type_score")}</option>
              </select>
            </Field>
            {q.type === "score" && <Field label={t("logit_q_min")}><input className="field" type="number" min="0" max="8" value={q.min} onChange={(e) => update(i, { min: e.target.value })} /></Field>}
            {q.type === "score" && <Field label={t("logit_q_max")}><input className="field" type="number" min="1" max="9" value={q.max} onChange={(e) => update(i, { max: e.target.value })} /></Field>}
          </div>
          <Field label={t("logit_q_instructions")}><input className="field" value={q.instructions} onChange={(e) => update(i, { instructions: e.target.value })} /></Field>
          {q.type === "choice" && <Field label={t("logit_q_options")}><textarea className="field" rows={3} value={q.options} onChange={(e) => update(i, { options: e.target.value })} /></Field>}
          <button type="button" className="btn btn-sm" onClick={() => setRubric((r) => r.filter((_, k) => k !== i))} disabled={rubric.length < 2}>{t("logit_remove")}</button>
        </div>
      ))}
      <button type="button" className="btn btn-sm" onClick={() => setRubric((r) => [...r, blankQuestion(r.length + 1)])}>{t("logit_add_question")}</button>
    </div>
  );
}

function TargetEditor({ targets, setTargets, servers, probes, probe, busy }) {
  const { t } = useApp();
  const update = (i, patch) => setTargets((r) => r.map((x, k) => (k === i ? { ...x, ...patch } : x)));
  return (
    <div className="space-y-2">
      {targets.map((x, i) => (
        <div key={i} className="panel space-y-2" data-testid="logit-target">
          <div className="grid gap-2 md:grid-cols-3">
            <Field label={t("logit_target_kind")}>
              <select className="field" value={x.kind} onChange={(e) => update(i, { kind: e.target.value })}>
                <option value="server">{t("logit_target_server")}</option>
                <option value="hf">{t("logit_target_hf")}</option>
              </select>
            </Field>
            {x.kind === "server" && (
              <Field label={t("logit_registered")}>
                <select className="field" value={x.model} onChange={(e) => update(i, { model: e.target.value, url: "" })}>
                  <option value="">{t("logit_other_server")}</option>
                  {servers.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select>
              </Field>
            )}
            {x.kind === "server" && !servers.some((m) => m.id === x.model) && (
              <>
                <Field label="URL"><input className="field" placeholder="http://127.0.0.1:8080" value={x.url} onChange={(e) => update(i, { url: e.target.value })} /></Field>
                <Field label={t("model")}><input className="field" value={x.model} onChange={(e) => update(i, { model: e.target.value })} /></Field>
              </>
            )}
          </div>
          {x.kind === "hf" && (
            <div className="grid gap-2 md:grid-cols-2">
              <Field label={t("logit_hf_base")}><input className="field" placeholder="D:\models\base" value={x.base_path} onChange={(e) => update(i, { base_path: e.target.value })} /></Field>
              <Field label={t("logit_hf_adapter")}><input className="field" placeholder="D:\adapters\lora" value={x.adapter_path} onChange={(e) => update(i, { adapter_path: e.target.value })} /></Field>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {x.kind === "server" && <Busy className="btn btn-sm" busy={busy[`probe${i}`]} onClick={() => probe(i, x)}>{t("logit_probe")}</Busy>}
            {probes[i] && (probes[i].supported
              ? <Chip className="chip-ok">{t("logit_probe_ok")}</Chip>
              : <span className="banner banner-warn">{probes[i].problem ? t.msg(probes[i].problem) : t("logit_probe_bad")}{probes[i].problem && t.hint(probes[i].problem) ? <span className="help block">{t("hint")}: {t.hint(probes[i].problem)}</span> : null}</span>)}
            <button type="button" className="btn btn-sm" onClick={() => setTargets((r) => r.filter((_, k) => k !== i))} disabled={targets.length < 2}>{t("logit_remove")}</button>
          </div>
        </div>
      ))}
      <button type="button" className="btn btn-sm" onClick={() => setTargets((r) => [...r, emptyTarget()])}>{t("logit_add_target")}</button>
    </div>
  );
}

export function LogitForm() {
  const { t, version, changed, notify } = useApp();
  const [rubric, setRubric] = useState([blankQuestion(1)]);
  const [items, setItems] = useState("");
  const [targets, setTargets] = useState([emptyTarget()]);
  const [result, setResult] = useState(null);
  const [probes, setProbes] = useState({});
  const [busy, act] = useBusy();
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const servers = useMemo(() => (modelsLoad.data?.models || []).filter((m) => m.kind === "server"), [modelsLoad.data]);

  const args = useMemo(() => ({
    rubric: rubric.map((q) => (q.type === "choice"
      ? { id: q.id, type: "choice", instructions: q.instructions, options: q.options.split("\n").map((s) => s.trim()).filter(Boolean) }
      : { id: q.id, type: "score", instructions: q.instructions, min: Number(q.min), max: Number(q.max) })),
    items: items.split(/^\s*---\s*$/m).map((s) => s.trim()).filter(Boolean).map((context, n) => ({ id: `i${n + 1}`, context })),
    targets: targets.map((x) => (x.kind === "hf"
      ? { kind: "hf", base_path: x.base_path.trim(), adapter_path: x.adapter_path.trim() }
      : (x.url.trim() ? { kind: "server", url: x.url.trim(), model: x.model.trim() } : x.model))),
  }), [rubric, items, targets]);

  const probe = (i, x) => act(`probe${i}`, async () => {
    const target = args.targets[i];
    const res = await api.call("logit_judge_probe", { targets: [target] });
    setProbes((p) => ({ ...p, [i]: res.servers[0] }));
  });
  const example = () => { setRubric(EXAMPLE.rubric); setItems(EXAMPLE.items); };
  const run = () => act("run", async () => {
    setResult(null);
    const out = await api.call("logit_judge_run", args);
    setResult(out);
    notify(t("logit_done"));
    changed();
  });
  const ready = args.items.length > 0 && args.rubric.every((q) => q.instructions) && args.targets.every((x) => (typeof x === "string" ? x : x.base_path || x.url));

  return (
    <div className="space-y-4" data-testid="logit-form">
      <p className="help">{t("logit_intro")}</p>
      <Section title={t("logit_rubric")} count={rubric.length} actions={<button type="button" className="btn btn-sm" onClick={example}>{t("logit_example")}</button>}>
        <RubricEditor rubric={rubric} setRubric={setRubric} />
      </Section>
      <Section title={t("logit_items")} count={args.items.length || undefined}>
        <Field label={t("logit_items_hint")}><textarea className="field" rows={6} value={items} onChange={(e) => setItems(e.target.value)} /></Field>
      </Section>
      <Section title={t("logit_targets")} count={targets.length}>
        <TargetEditor targets={targets} setTargets={setTargets} servers={servers} probes={probes} probe={probe} busy={busy} />
        <p className="help">{t("logit_targets_help")}</p>
      </Section>
      <div className="flex flex-wrap items-center gap-3">
        <Busy className="btn btn-primary" busy={busy.run} disabled={!ready} onClick={run}><Icon d={ICONS.run} size={14} />{busy.run ? t("logit_running") : t("logit_run")}</Busy>
      </div>
      {result && <><div className="help">{result.label}</div><LogitResult data={result} /></>}
    </div>
  );
}
