import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { num, pct } from "../format.js";

const ARMS = ["generic", "current", "candidate"];
const TEMPLATE = JSON.stringify({
  suit: "repository-intake",
  name: "",
  cases: [
    { id: "k1", title: "Caso de construcción", used_for_building: true, checks: [{ kind: "file_exists", path: "out/report.md" }] },
    { id: "k2", title: "Caso de juicio", checks: [{ kind: "file_exists", path: "out/report.md" }, { kind: "regex", path: "out/report.md", pattern: "app" }, { kind: "judge", rubric: "Cita una capacidad del árbol" }] },
  ],
  candidate: { description: "", built_from_cases: ["k1"] },
}, null, 2);

const mean = (s, lang, digits = 1) => (s && s.n ? `${num(s.mean, digits, lang)}${s.sd != null ? ` ± ${num(s.sd, digits, lang)}` : ""} (${s.n})` : "—");

function Score({ score }) {
  const { t, lang } = useApp();
  return (
    <div className="space-y-4">
      {score.warnings.map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
      <Section title={t("suit_arms")}>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table className="tbl">
            <thead><tr>
              <th>{t("suit_col_arm")}</th><th>{t("suit_col_cases")}</th><th>{t("suit_col_success")}</th><th>{t("suit_col_time")}</th>
              <th>{t("suit_col_correction")}</th><th>{t("suit_col_defects")}</th><th>{t("suit_col_tokens")}</th>
            </tr></thead>
            <tbody>
              {ARMS.map((arm) => {
                const a = score.aggregate[arm];
                return (
                  <tr key={arm}>
                    <td><b>{t(`suit_arm_${arm}`)}</b></td>
                    <td>{a.cases}{a.pending ? <span className="help"> +{a.pending} {t("suit_pending")}</span> : null}{a.missing ? <span className="help"> · {a.missing} {t("suit_missing")}</span> : null}</td>
                    <td>{a.success_rate == null ? "—" : <>{pct(a.success_rate, 0, lang)} <span className="help">[{pct(a.success_ci95[0], 0, lang)}–{pct(a.success_ci95[1], 0, lang)}]</span></>}</td>
                    <td>{mean(a.time_s, lang)}</td><td>{mean(a.correction_minutes, lang)}</td><td>{mean(a.defects, lang)}</td><td>{mean(a.tokens, lang, 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="help">{t("suit_spread_help")}</p>
        </div>
      </Section>
      <Section title={t("suit_paired")}>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table className="tbl">
            <thead><tr><th>{t("suit_pair")}</th><th>{t("suit_col_cases")}</th><th>{t("suit_better")}</th><th>{t("suit_worse")}</th><th>{t("suit_same")}</th><th>{t("suit_sign_p")}</th></tr></thead>
            <tbody>
              {Object.entries(score.paired).map(([k, p]) => {
                const [a, b] = k.split("_vs_");
                return <tr key={k}><td>{t(`suit_arm_${a}`)} / {t(`suit_arm_${b}`)}</td><td>{p.cases}</td><td>{p.better}</td><td>{p.worse}</td><td>{p.same}</td><td>{p.sign_test_p == null ? "—" : num(p.sign_test_p, 3, lang)}</td></tr>;
              })}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title={t("suit_per_case")} count={score.cases_total}>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table className="tbl">
            <thead><tr><th>{t("suit_case")}</th>{ARMS.map((a) => <th key={a}>{t(`suit_arm_${a}`)}</th>)}</tr></thead>
            <tbody>
              {score.per_case.map((c) => (
                <tr key={c.case}>
                  <td><b>{c.case}</b> <span className="help">{c.title}</span> {c.excluded ? <Chip className="chip-amber" title={t("suit_excluded_help")}>{t("suit_excluded")}</Chip> : null}</td>
                  {ARMS.map((arm) => {
                    const r = c.arms[arm];
                    return <td key={arm}>{!r ? <span className="help">—</span> : r.success == null ? <Chip>{t("suit_waiting")}</Chip> : <Chip className={r.success >= 1 ? "chip-ok" : r.success <= 0 ? "chip-danger" : "chip-amber"}>{pct(r.success, 0, lang)}</Chip>}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function Correction({ detail, onDone }) {
  const { t, notify } = useApp();
  const cases = detail.evaluation.definition.cases;
  const [c, setC] = useState(cases[0]?.id || "");
  const [arm, setArm] = useState("current");
  const [minutes, setMinutes] = useState("");
  const [busy, run] = useBusy();
  const save = () => run("save", async () => {
    await api.call("pygmalion_suit_eval_record", { id: detail.evaluation.id, case: c, arm, correction_minutes: Number(minutes) });
    setMinutes("");
    notify(t("suit_saved"));
    onDone();
  });
  return (
    <Section title={t("suit_correction_title")}>
      <div className="panel space-y-2">
        <p className="help">{t("suit_correction_help")}</p>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("suit_case")}><select className="field" value={c} onChange={(e) => setC(e.target.value)}>{cases.map((x) => <option key={x.id} value={x.id}>{x.id}</option>)}</select></Field>
          <Field label={t("suit_col_arm")}><select className="field" value={arm} onChange={(e) => setArm(e.target.value)}>{ARMS.map((a) => <option key={a} value={a}>{t(`suit_arm_${a}`)}</option>)}</select></Field>
          <Field label={t("suit_minutes")}><input className="field" type="number" min="0" step="0.5" value={minutes} onChange={(e) => setMinutes(e.target.value)} style={{ width: 110 }} /></Field>
          <Busy className="btn btn-primary" busy={busy.save} disabled={minutes === ""} onClick={save}><Icon d={ICONS.check} size={14} />{t("suit_save")}</Busy>
        </div>
      </div>
    </Section>
  );
}

function Detail({ id, onGone }) {
  const { t, notify, confirm } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("pygmalion_suit_eval_get", { id }), [id]);
  const [busy, run] = useBusy();
  const exportIt = (mode) => run("export", async () => {
    const out = await api.call("pygmalion_suit_eval_export", { id, mode });
    notify(out.exported.delivered ? t("suit_sent") : t("suit_file", { file: out.exported.file }));
    reload();
  });
  const remove = () => run("delete", async () => {
    if (!(await confirm({ title: t("suit_delete"), message: t("suit_delete_confirm"), confirmLabel: t("suit_delete"), danger: true }))) return;
    await api.call("pygmalion_suit_eval_delete", { id, confirm: true });
    onGone();
  });
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Spinner />;
  const ev = data.evaluation;
  const ex = ev.exported || {};
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ margin: 0 }}>{ev.name} <span className="help mono">{ev.suit}</span></h2>
        {ex.delivered ? <Chip className="chip-ok">{t("suit_exported_api")} <Rel ts={ex.at} /></Chip> : ex.file ? <Chip className="chip-amber" title={ex.file}>{t("suit_exported_file")}</Chip> : <Chip>{t("suit_not_exported")}</Chip>}
        <Busy className="btn btn-primary" busy={busy.export} onClick={() => exportIt("auto")} disabled={data.score.cases_scored < 1}><Icon d={ICONS.publish} size={14} />{t("suit_export")}</Busy>
        <Busy className="btn" busy={busy.export} onClick={() => exportIt("file")} disabled={data.score.cases_scored < 1}>{t("suit_export_file")}</Busy>
        <Busy className="btn btn-danger" busy={busy.delete} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("suit_delete")}</Busy>
      </div>
      {ev.definition.candidate.description ? <p className="help">{t("suit_candidate")}: {ev.definition.candidate.description}</p> : null}
      <Score score={data.score} />
      <Correction detail={data} onDone={reload} />
    </div>
  );
}

export default function Trajes() {
  const { t, version, notify } = useApp();
  const list = useLoad(() => api.call("pygmalion_suit_eval_list", {}), [version]);
  const [sel, setSel] = useState(null);
  const [draft, setDraft] = useState(TEMPLATE);
  const [busy, run] = useBusy();
  const create = () => run("create", async () => {
    let body;
    try { body = JSON.parse(draft); } catch { notify(t("suit_bad_json")); return; }
    const out = await api.call("pygmalion_suit_eval_define", body);
    setSel(out.evaluation.id);
    list.reload();
  });
  const rows = list.data?.evaluations || [];
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("suit_title")}</h1>
        <Busy className="btn" busy={false} onClick={list.reload}><Icon d={ICONS.refresh} size={14} />{t("reload")}</Busy>
      </div>
      <p className="help">{t("suit_help")}</p>
      <ErrorBox error={list.error} />
      <Section title={t("suit_list")} count={rows.length}>
        {list.loading && !list.data && <Spinner />}
        {!list.loading && !rows.length && <Empty>{t("suit_none")}</Empty>}
        <div className="space-y-2">
          {rows.map((r) => (
            <button key={r.id} type="button" className="panel" onClick={() => setSel(r.id)} style={{ textAlign: "left", width: "100%", outline: sel === r.id ? "2px solid var(--accent)" : "none" }}>
              <b>{r.name}</b> <span className="help mono">{r.suit}</span>
              <span className="help"> · {t("suit_scored", { n: r.cases_scored, total: r.cases })}</span>
              {r.excluded.length ? <Chip className="chip-amber">{t("suit_excluded")}: {r.excluded.join(", ")}</Chip> : null}
              {r.exported?.delivered ? <Chip className="chip-ok">{t("suit_exported_api")}</Chip> : r.exported?.file ? <Chip className="chip-amber">{t("suit_exported_file")}</Chip> : null}
            </button>
          ))}
        </div>
      </Section>
      {sel && <Detail key={sel} id={sel} onGone={() => { setSel(null); list.reload(); }} />}
      <Section title={t("suit_define")}>
        <div className="panel space-y-2">
          <p className="help">{t("suit_define_help")}</p>
          <textarea className="field mono" rows={14} value={draft} onChange={(e) => setDraft(e.target.value)} spellCheck={false} aria-label={t("suit_define")} />
          <Busy className="btn btn-primary" busy={busy.create} onClick={create}><Icon d={ICONS.plus} size={14} />{t("suit_define_go")}</Busy>
        </div>
      </Section>
    </div>
  );
}
