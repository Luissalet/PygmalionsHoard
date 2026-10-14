import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { compact, num, shortClock } from "../format.js";
import { usePoll } from "../components/hooks.js";
import { Busy, Check, Chip, CopyButton, Drawer, Empty, ErrorBox, Field, Icon, ICONS, KV, PageHead, Rel, Section, Seg, useBusy, useLoad } from "../components/ui.jsx";

const TYPES = ["auto", "llama_server", "vllm"];

function UrlRow({ label, value }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="help" style={{ minWidth: 170 }}>{label}</span>
      <code className="mono min-w-0 flex-1">{value}</code>
      <CopyButton text={value} />
    </div>
  );
}

function ProxyPanel({ status, reload }) {
  const { t, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [draft, setDraft] = useState(null);
  const [probe, setProbe] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (status && !draft) setDraft({ url: status.upstream_url || "", type: status.upstream_type || "auto", top: status.top_logprobs || 1, faithful: !!status.faithful_sampling });
  }, [status, draft]);
  if (!status || !draft) return <div className="help">{t("loading")}</div>;
  const set = (k, v) => setDraft((d) => ({ ...d, [k]: v }));
  const args = { upstream_url: draft.url.trim(), upstream_type: draft.type, top_logprobs: Number(draft.top) || 1, faithful_sampling: draft.faithful };
  const act = (action, key) => run(key, async () => {
    setError(null);
    try {
      const r = await api.call("capture_proxy", { action, ...args });
      if (r.upstream) setProbe(r.upstream);
      notify(action === "start" ? t("cap_started") : action === "stop" ? t("cap_stopped") : action === "check" ? t("cap_checked") : t("saved"));
      changed();
      reload();
    } catch (e) {
      setError(e);
    }
  });
  const on = !!status.enabled;
  return (
    <Section id="cap-proxy" title={t("cap_proxy")} actions={<Chip className={on ? "chip-ok" : "chip-amber"}>{on ? t("cap_on") : t("cap_off")}</Chip>}>
      <div className="panel space-y-3">
        <p className="help">{t("cap_proxy_help")}</p>
        <ErrorBox error={error} />
        <div className="grid gap-3 sm:grid-cols-[2fr_1fr_90px]">
          <Field label={t("cap_upstream")} hint={t("cap_upstream_hint")}><input className="field mono" value={draft.url} onChange={(e) => set("url", e.target.value)} placeholder="http://127.0.0.1:8093" /></Field>
          <Field label={t("cap_type")}>
            <select className="field" value={draft.type} onChange={(e) => set("type", e.target.value)}>
              {TYPES.map((v) => <option key={v} value={v}>{t(`cap_type_${v}`)}</option>)}
            </select>
          </Field>
          <Field label={t("cap_top")}><input className="field" type="number" min="1" max="20" value={draft.top} onChange={(e) => set("top", e.target.value)} /></Field>
        </div>
        <div>
          <Check checked={draft.faithful} onChange={(v) => set("faithful", v)}>{t("cap_faithful")}</Check>
          <p className="help ml-6">{t("cap_faithful_help")}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Busy className="btn btn-primary" busy={busy.start} disabled={!draft.url.trim()} onClick={() => act("start", "start")}><Icon d={ICONS.play} size={13} />{t("cap_start")}</Busy>
          <Busy className="btn" busy={busy.stop} disabled={!on} onClick={() => act("stop", "stop")}><Icon d={ICONS.stop} size={13} />{t("cap_stop")}</Busy>
          <Busy className="btn" busy={busy.configure} onClick={() => act("configure", "configure")}>{t("save")}</Busy>
          <Busy className="btn" busy={busy.check} disabled={!draft.url.trim()} onClick={() => act("check", "check")}><Icon d={ICONS.refresh} size={13} />{t("cap_check")}</Busy>
        </div>
        {probe && (
          <div className={`banner ${probe.reachable ? "banner-info" : "banner-warn"}`} role="status">
            {probe.configured === false ? t("cap_no_upstream") : probe.reachable
              ? t("cap_reachable", { type: probe.detected, detail: probe.detail || "", models: (probe.models || []).join(", ") || "—" })
              : t("cap_unreachable", { detail: probe.detail || "" })}
          </div>
        )}
        <div className="space-y-1.5">
          <h3>{t("cap_urls")}</h3>
          <UrlRow label={t("cap_url_base")} value={status.base_url} />
          <UrlRow label={t("cap_url_session")} value={status.session_url} />
          <UrlRow label={t("cap_url_harness")} value={status.harness_session_url} />
          <p className="help">{t("cap_urls_help")}</p>
        </div>
        <div className="kpis">
          <div className="kpi"><b className="num">{num(status.counts?.sessions)}</b><span>{t("cap_sessions")}</span></div>
          <div className="kpi"><b className="num">{num(status.counts?.calls)}</b><span>{t("cap_calls")}</span></div>
          <div className="kpi"><b className="num">{num(status.counts?.unverified)}</b><span>{t("cap_unverified")}</span></div>
        </div>
      </div>
    </Section>
  );
}

function VerifiedChip({ ok, note }) {
  const { t } = useApp();
  return <Chip className={ok ? "chip-ok" : "chip-amber"} title={note || undefined}>{ok ? t("cap_verified") : t("cap_unverified_one")}</Chip>;
}

function Sessions({ harnesses, enabled }) {
  const { t, lang, version } = useApp();
  const [harness, setHarness] = useState("");
  const { data, error, reload } = useLoad(() => api.call("capture_sessions", { harness, limit: 100 }), [harness, version]);
  usePoll(reload, enabled ? 4000 : 0);
  const rows = data?.sessions || [];
  const names = [...new Set([...(harnesses || []).map((h) => h.harness), ...(data?.harnesses || []).map((h) => h.harness)])];
  return (
    <Section id="cap-sessions" title={t("cap_sessions")} count={rows.length}
      actions={<select className="field" style={{ width: "auto" }} value={harness} onChange={(e) => setHarness(e.target.value)} aria-label={t("cap_harness")}>
        <option value="">{t("cap_all_harnesses")}</option>
        {names.map((h) => <option key={h || "-"} value={h}>{h || t("cap_no_harness")}</option>)}
      </select>}>
      <ErrorBox error={error} />
      {data && !rows.length ? <Empty>{t("cap_no_sessions")}</Empty> : (
        <div className="scroll-x panel panel-tight">
          <table>
            <thead><tr><th>{t("cap_session")}</th><th>{t("cap_harness")}</th><th className="r">{t("cap_calls")}</th><th className="r">{t("cap_paths")}</th><th className="r">{t("cap_sampled")}</th><th>{t("cap_unverified")}</th><th>{t("cap_reward")}</th><th>{t("updated")}</th></tr></thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.id}>
                  <td><a href={`#/captura/${encodeURIComponent(s.id)}`} className="btn-link mono">{s.id}</a><div className="help">{s.model}</div></td>
                  <td>{s.harness || "—"}</td>
                  <td className="r num">{num(s.calls)}</td>
                  <td className="r num" title={t("cap_paths_help", { roots: s.roots, leaves: s.leaves })}>{num(s.leaves)}</td>
                  <td className="r num">{compact(s.completion_tokens, lang)}</td>
                  <td>{s.unverified ? <Chip className="chip-amber">{s.unverified}</Chip> : <span className="help">0</span>}</td>
                  <td>{s.reward === null || s.reward === undefined ? <span className="help">—</span> : <Chip className={s.reward > 0 ? "chip-ok" : ""}>{num(s.reward, 2, lang)}{s.outcome ? ` · ${s.outcome}` : ""}</Chip>}</td>
                  <td><Rel ts={s.updated_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

function ExportPanel({ harnesses }) {
  const { t, lang, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [f, setF] = useState({
    harness: [], require_reward: false, min_reward: "", include_unverified: false, loss_on: "all", dedupe_shared: true, balance: "none", cap: "100",
    since: "", until: "", sft: true, rl: true, dataset: false, dataset_name: "", seed: 42,
  });
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);
  const set = (k, v) => setF((x) => ({ ...x, [k]: v }));
  const toggleHarness = (h) => set("harness", f.harness.includes(h) ? f.harness.filter((x) => x !== h) : [...f.harness, h]);
  const args = (dry) => {
    const formats = [f.dataset && "dataset", f.sft && "sft", f.rl && "rl"].filter(Boolean);
    const a = {
      harness: f.harness, require_reward: f.require_reward, include_unverified: f.include_unverified, loss_on: f.loss_on, dedupe_shared: f.dedupe_shared,
      seed: Number(f.seed) || 0, dry_run: dry, formats,
    };
    if (f.min_reward !== "") a.min_reward = Number(f.min_reward);
    if (f.balance === "min") a.balance_by_harness = "min";
    if (f.balance === "cap") a.balance_by_harness = Math.max(1, Number(f.cap) || 1);
    if (f.since) a.since = f.since;
    if (f.until) a.until = f.until;
    if (f.dataset) a.dataset_name = f.dataset_name.trim();
    return a;
  };
  const go = (dry) => run(dry ? "dry" : "go", async () => {
    setError(null);
    try {
      const r = await api.call("capture_export", args(dry));
      setOut(r);
      if (!dry) { notify(t("cap_exported", { n: r.rows })); changed(); }
    } catch (e) {
      setError(e);
    }
  });
  const names = (harnesses || []).map((h) => h.harness).filter((h) => h);
  const nothing = !f.sft && !f.rl && !f.dataset;
  return (
    <Section id="cap-export" title={t("cap_export")}>
      <div className="panel space-y-3">
        <p className="help">{t("cap_export_help")}</p>
        <ErrorBox error={error} />
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <h3>{t("cap_filters")}</h3>
            {names.length > 0 && <div className="flex flex-wrap gap-3">{names.map((h) => <Check key={h} checked={f.harness.includes(h)} onChange={() => toggleHarness(h)}>{h}</Check>)}</div>}
            {names.length > 0 && <p className="help">{t("cap_harness_all")}</p>}
            <div className="grid grid-cols-2 gap-2">
              <Field label={t("cap_since")}><input className="field" type="date" value={f.since} onChange={(e) => set("since", e.target.value)} /></Field>
              <Field label={t("cap_until")}><input className="field" type="date" value={f.until} onChange={(e) => set("until", e.target.value)} /></Field>
            </div>
            <Check checked={f.require_reward} onChange={(v) => set("require_reward", v)}>{t("cap_require_reward")}</Check><br />
            <label className="check">{t("cap_min_reward")}<input className="field" style={{ width: 80 }} type="number" step="0.1" value={f.min_reward} onChange={(e) => set("min_reward", e.target.value)} /></label><br />
            <Check checked={f.include_unverified} onChange={(v) => set("include_unverified", v)}>{t("cap_include_unverified")}</Check>
          </div>
          <div className="space-y-1.5">
            <h3>{t("cap_balance")}</h3>
            <Seg value={f.balance} onChange={(v) => set("balance", v)} label={t("cap_balance")}
              options={[{ value: "none", label: t("cap_balance_none") }, { value: "min", label: t("cap_balance_min") }, { value: "cap", label: t("cap_balance_cap") }]} />
            {f.balance === "cap" && <Field label={t("cap_balance_cap_n")}><input className="field" style={{ width: 100 }} type="number" min="1" value={f.cap} onChange={(e) => set("cap", e.target.value)} /></Field>}
            <h3>{t("cap_loss")}</h3>
            <Seg value={f.loss_on} onChange={(v) => set("loss_on", v)} label={t("cap_loss")} options={[{ value: "all", label: t("cap_loss_all") }, { value: "last", label: t("cap_loss_last") }]} />
            <Check checked={f.dedupe_shared} onChange={(v) => set("dedupe_shared", v)}>{t("cap_dedupe")}</Check>
            <h3>{t("cap_formats")}</h3>
            <Check checked={f.sft} onChange={(v) => set("sft", v)}>{t("cap_fmt_sft")}</Check><br />
            <Check checked={f.rl} onChange={(v) => set("rl", v)}>{t("cap_fmt_rl")}</Check><br />
            <Check checked={f.dataset} onChange={(v) => set("dataset", v)}>{t("cap_fmt_dataset")}</Check>
            {f.dataset && <Field label={t("cap_dataset_name")} hint={t("cap_dataset_hint")}><input className="field" value={f.dataset_name} onChange={(e) => set("dataset_name", e.target.value)} placeholder="Faustus capturado" /></Field>}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Busy className="btn" busy={busy.dry} onClick={() => go(true)}><Icon d={ICONS.search} size={13} />{t("cap_dry")}</Busy>
          <Busy className="btn btn-primary" busy={busy.go} disabled={nothing || (f.dataset && !f.dataset_name.trim())} onClick={() => go(false)}><Icon d={ICONS.download} size={13} />{t("cap_do_export")}</Busy>
        </div>
        {out && <ExportResult out={out} lang={lang} />}
      </div>
    </Section>
  );
}

function ExportResult({ out, lang }) {
  const { t } = useApp();
  const rep = out.report || {};
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Chip className={out.dry_run ? "chip-info" : "chip-ok"}>{out.dry_run ? t("cap_result_dry") : t("cap_result_done")}</Chip>
        <Chip>{t("cap_rows", { n: out.rows, total: out.rows_available })}</Chip>
        <Chip>{t("cap_loss_tokens", { n: compact(out.loss_tokens, lang) })}</Chip>
        <Chip>{t("cap_with_reward", { n: out.with_reward })}</Chip>
        {rep.unverified_excluded > 0 && <Chip className="chip-amber">{t("cap_unverified_excluded", { n: rep.unverified_excluded })}</Chip>}
      </div>
      <div className="scroll-x">
        <table>
          <thead><tr><th>{t("cap_harness")}</th><th className="r">{t("cap_available")}</th><th className="r">{t("cap_kept")}</th><th className="r">{t("cap_loss_tokens_col")}</th></tr></thead>
          <tbody>
            {Object.entries(out.per_harness || {}).map(([h, v]) => (
              <tr key={h || "-"}><td>{h || t("cap_no_harness")}</td><td className="r num">{num(v.available)}</td><td className="r num">{num(v.kept)}</td><td className="r num">{compact(v.loss_tokens_kept, lang)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      {!out.dry_run && (
        <KV items={[[t("cap_folder"), <code key="f" className="mono">{out.folder}</code>],
          ...Object.entries(out.files || {}).map(([k, v]) => [k, <code key={k} className="mono">{v}</code>]),
          [t("nav_datasets"), out.dataset ? <a key="d" className="btn-link" href={`#/datasets/${encodeURIComponent(out.dataset.id)}`}>{out.dataset.name} · v{out.version?.n}</a> : null]]} />
      )}
    </div>
  );
}

function OutcomeForm({ session, reload }) {
  const { t, notify, changed } = useApp();
  const [busy, run] = useBusy();
  const [reward, setReward] = useState(session.reward === null || session.reward === undefined ? "" : String(session.reward));
  const [outcome, setOutcome] = useState(session.outcome || "");
  const [note, setNote] = useState(session.note || "");
  const save = (clear) => run("o", async () => {
    await api.call("capture_outcome", clear ? { session: session.id, clear: true } : { session: session.id, reward: reward === "" ? null : Number(reward), outcome: outcome || null, note });
    if (clear) { setReward(""); setOutcome(""); setNote(""); }
    notify(t("saved"));
    changed();
    reload();
  });
  return (
    <div className="panel space-y-2">
      <h3>{t("cap_outcome")}</h3>
      <p className="help">{t("cap_outcome_help")}</p>
      <div className="grid gap-3 sm:grid-cols-[110px_160px_1fr_auto] sm:items-end">
        <Field label={t("cap_reward")}><input className="field" type="number" step="0.1" value={reward} onChange={(e) => setReward(e.target.value)} /></Field>
        <Field label={t("cap_outcome_label")}><input className="field" value={outcome} onChange={(e) => setOutcome(e.target.value)} placeholder="success" /></Field>
        <Field label={t("notes")}><input className="field" value={note} onChange={(e) => setNote(e.target.value)} /></Field>
        <div className="flex gap-2">
          <Busy className="btn btn-primary" busy={busy.o} onClick={() => save(false)}>{t("save")}</Busy>
          <button type="button" className="btn" disabled={busy.o} onClick={() => save(true)}>{t("cap_clear")}</button>
        </div>
      </div>
    </div>
  );
}

function tokenShade(lp) {
  if (lp === null || lp === undefined) return undefined;
  const pct = Math.min(60, Math.round((-lp / 6) * 100));
  return pct > 4 ? { background: `color-mix(in srgb, var(--danger) ${pct}%, transparent)` } : undefined;
}

function NodeDrawer({ session, id, onClose }) {
  const { t, lang } = useApp();
  const { data, error } = useLoad(() => api.call("capture_session", { session, node: id, max_tokens: 400 }), [session, id]);
  const n = data?.node;
  return (
    <Drawer title={t("cap_call_title", { id })} onClose={onClose}>
      <ErrorBox error={error} />
      {!n ? <div className="help">{t("loading")}</div> : (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <VerifiedChip ok={n.verified} note={n.verify_note} />
            <Chip>{n.prompt_ids_source || "—"}</Chip>
            <Chip>{n.upstream_type}</Chip>
            <Chip>{n.finish_reason || "—"}</Chip>
            {n.flags?.eos_appended && <Chip className="chip-amber">{t("cap_eos_appended")}</Chip>}
          </div>
          {n.verify_note && <div className="banner banner-warn">{n.verify_note}</div>}
          <KV items={[[t("cap_prompt_tokens"), `${num(n.prompt_tokens)} (${t("cap_new_tokens", { n: num(n.new_prompt_tokens) })})`], [t("cap_sampled"), num(n.completion_tokens)],
            [t("cap_model"), n.model], [t("cap_parent"), n.parent || "—"], [t("cap_latency"), n.latency_ms ? `${num(n.latency_ms)} ms` : ""],
            [t("cap_sampling"), Object.keys(n.sampling || {}).length ? <code key="s" className="mono">{JSON.stringify(n.sampling)}</code> : ""],
            [t("cap_lp_mean"), n.flags?.lp_mean !== undefined ? num(n.flags.lp_mean, 3, lang) : ""], [t("cap_lp_min"), n.flags?.lp_min !== undefined ? num(n.flags.lp_min, 3, lang) : ""],
            [t("cap_reward"), n.reward === null || n.reward === undefined ? "" : num(n.reward, 2, lang)]]} />
          <h3>{t("cap_tokens_title", { shown: n.tokens_shown, total: n.completion_tokens })}</h3>
          <div className="logbox" aria-label={t("cap_tokens_title", { shown: n.tokens_shown, total: n.completion_tokens })}>
            {n.tokens.map((x) => <span key={x.i} style={tokenShade(x.logprob)} title={`#${x.id}  ${x.logprob === null || x.logprob === undefined ? "—" : num(x.logprob, 3, lang)}`}>{x.token ?? `[${x.id}]`}</span>)}
          </div>
          {n.lowest?.length > 0 && (
            <>
              <h3>{t("cap_lowest")}</h3>
              <div className="flex flex-wrap gap-1.5">{n.lowest.map((x) => <Chip key={x.i} title={`#${x.id}`}>{JSON.stringify(x.token ?? `#${x.id}`)} {num(x.logprob, 2, lang)}</Chip>)}</div>
            </>
          )}
          <h3>{t("cap_messages")} <span className="chip">{n.messages.length}</span></h3>
          <div className="space-y-1.5">
            {n.messages.slice(-6).map((m, i) => (
              <div key={i} className="panel panel-tight"><div className="help">{m.role}{m.tool_calls ? ` · ${t("cap_tool_calls")}` : ""}</div><div className="clamp2" style={{ WebkitLineClamp: 6 }}>{typeof m.content === "string" ? m.content : JSON.stringify(m.content)}</div></div>
            ))}
            {n.messages.length > 6 && <div className="help">{t("cap_older_hidden", { n: n.messages.length - 6 })}</div>}
          </div>
          <h3>{t("cap_response")}</h3>
          <div className="logbox">{n.response ? (n.response.content || "") + (n.response.tool_calls ? "\n" + JSON.stringify(n.response.tool_calls, null, 1) : "") : "—"}</div>
          {n.tools?.length > 0 && <div className="help">{t("cap_tools", { names: n.tools.map((x) => x.function?.name).filter(Boolean).join(", ") })}</div>}
        </div>
      )}
    </Drawer>
  );
}

function SessionDetail({ id }) {
  const { t, lang, version } = useApp();
  const [open, setOpen] = useState("");
  const { data, error, reload } = useLoad(() => api.call("capture_session", { session: id }), [id, version]);
  usePoll(reload, 5000);
  if (!data) return <div className="space-y-3"><a href="#/captura" className="btn-link">{t("cap_back")}</a>{error ? <ErrorBox error={error} /> : <div className="help">{t("loading")}</div>}</div>;
  const s = data.session;
  return (
    <div className="space-y-5">
      <PageHead title={s.id} subtitle={t("cap_session_sub", { harness: s.harness || t("cap_no_harness"), calls: s.calls, paths: s.leaves })} actions={<a href="#/captura" className="btn">{t("cap_back")}</a>} />
      <div className="flex flex-wrap gap-2">
        <Chip>{s.model || "—"}</Chip>
        <Chip>{t("cap_roots", { n: s.roots })}</Chip>
        {s.unverified > 0 && <Chip className="chip-amber">{t("cap_unverified_excluded", { n: s.unverified })}</Chip>}
        <Chip>{t("cap_sampled_n", { n: compact(s.completion_tokens, lang) })}</Chip>
      </div>
      <OutcomeForm key={`${s.reward}|${s.outcome}|${s.note}`} session={s} reload={reload} />
      <Section id="cap-tree" title={t("cap_tree")} count={data.nodes.length}>
        <p className="help">{t("cap_tree_help")}</p>
        {data.truncated && <div className="banner banner-info">{t("cap_truncated")}</div>}
        <div className="scroll-x panel panel-tight">
          <table>
            <thead><tr><th className="r">#</th><th>{t("cap_call")}</th><th className="r">{t("cap_prompt_tokens")}</th><th className="r">{t("cap_sampled")}</th><th className="r">{t("cap_lp_mean")}</th><th>{t("state")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {data.nodes.map((n) => (
                <tr key={n.id} style={{ cursor: "pointer" }} onClick={() => setOpen(n.id)}>
                  <td className="r num">{n.seq}</td>
                  <td style={{ overflowWrap: "anywhere", maxWidth: 520 }}>
                    <div style={{ paddingLeft: Math.min(n.depth, 12) * 14, borderLeft: n.depth ? "2px solid var(--line)" : undefined, marginLeft: n.depth ? 4 : 0 }}>
                      <button type="button" className="btn-link mono" onClick={(e) => { e.stopPropagation(); setOpen(n.id); }}>{n.id.slice(-8)}</button>
                      {n.children > 1 && <Chip className="chip-info">{t("cap_branches", { n: n.children })}</Chip>}
                      {n.parent === null && n.seq > 1 && <Chip className="chip-amber" title={t("cap_new_root_help")}>{t("cap_new_root")}</Chip>}
                      <div className="help clamp2">{n.input ? `${t("cap_in")}: ${n.input}` : ""}</div>
                      <div className="clamp2">{n.tool_calls.length ? `${t("cap_tool_calls")}: ${n.tool_calls.join(", ")}` : n.output}</div>
                    </div>
                  </td>
                  <td className="r num">{num(n.prompt_tokens)}</td>
                  <td className="r num">{num(n.completion_tokens)}</td>
                  <td className="r num">{n.lp_mean === null || n.lp_mean === undefined ? "—" : num(n.lp_mean, 2, lang)}</td>
                  <td><VerifiedChip ok={n.verified} note={n.verify_note} /> {n.finish_reason && n.finish_reason !== "stop" && <Chip>{n.finish_reason}</Chip>}</td>
                  <td className="help" style={{ whiteSpace: "nowrap" }}>{shortClock(n.ts, lang)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      {open && <NodeDrawer session={id} id={open} onClose={() => setOpen("")} />}
    </div>
  );
}

export default function Captura({ param }) {
  const { t, version } = useApp();
  const status = useLoad(() => api.call("capture_proxy", { action: "status" }), [version]);
  usePoll(status.reload, 6000);
  if (param) return <SessionDetail id={param} />;
  return (
    <div className="space-y-6">
      <PageHead title={t("nav_capture")} subtitle={t("cap_sub")} />
      <ErrorBox error={status.error} />
      <ProxyPanel status={status.data} reload={status.reload} />
      <Sessions harnesses={status.data?.harnesses} enabled={!!status.data?.enabled} />
      <ExportPanel harnesses={status.data?.harnesses} />
    </div>
  );
}
