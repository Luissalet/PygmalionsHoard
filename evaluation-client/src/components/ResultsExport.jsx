import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { answers } from "../copilot-context.js";
import { ErrorBox } from "./ui.jsx";

export default function ResultsExport({ runId, model, onlyFailed, modelName }) {
  const { t, lang } = useApp();
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const sequence = useRef(0);
  const identity = JSON.stringify([runId, model, onlyFailed]);
  const current = useRef(identity);
  current.current = identity;
  useEffect(() => {
    sequence.current += 1;
    setResult(null);
    setError(null);
    setPending(false);
    return () => { sequence.current += 1; };
  }, [identity]);
  const prepare = async () => {
    const request = ++sequence.current;
    const criteria = identity;
    setPending(true);
    setResult(null);
    setError(null);
    try {
      const data = await api.call("run_results_export", { run: runId, model, only_failed: onlyFailed });
      if (sequence.current === request && current.current === criteria) setResult({ identity: criteria, data });
    } catch (e) {
      if (sequence.current === request && current.current === criteria) setError(e);
    } finally {
      if (sequence.current === request && current.current === criteria) setPending(false);
    }
  };
  const data = result?.identity === identity ? result.data : null;
  useEffect(() => {
    const provide = (event) => {
      if (!answers(event, runId)) return;
      event.detail.provide({ kind: "run-results-export", dirty: false, fields: {
        run: runId, model, only_failed: onlyFailed, preparing: pending,
        prepared: data ? { count: data.count, read_started_at: data.read_started_at, read_finished_at: data.read_finished_at } : null,
      } });
    };
    window.addEventListener("galton:context-request", provide);
    return () => window.removeEventListener("galton:context-request", provide);
  }, [runId, model, onlyFailed, pending, data]);
  const download = (format) => {
    const blob = new Blob([data[`${format}_text`]], { type: format === "csv" ? "text/csv;charset=utf-8" : "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `results-${runId}.${format}`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const date = (value) => new Date(value).toLocaleString(lang === "en" ? "en-GB" : "es-ES");
  return (
    <div className="panel space-y-2" aria-label={t("results_export")}>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="mr-auto">{t("results_export")}</h3>
        <button type="button" className="btn" disabled={pending} onClick={prepare}>{pending ? t("results_preparing") : t("results_prepare")}</button>
      </div>
      <p className="help">{t("results_export_scope", { model: modelName || t("all_models"), filter: onlyFailed ? t("only_failed") : t("results_all_states") })}</p>
      <p className="help">{t("results_export_help")}</p>
      <ErrorBox error={error} />
      {data && <div className="space-y-2" role="status">
        <p>{t("results_export_count", { n: data.count })}</p>
        <p className="help" style={{ overflowWrap: "anywhere" }}>{t("results_export_interval", { start: date(data.read_started_at), end: date(data.read_finished_at) })}</p>
        <p className="help">{t("results_export_state")} <span>{t(`state_${data.state_at_start}`)}</span>. {t("results_export_live")}</p>
        {!data.count && <p className="help">{t("results_export_empty")}</p>}
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn" onClick={() => download("csv")}>{t("results_download_csv")}</button>
          <button type="button" className="btn" onClick={() => download("json")}>{t("results_download_json")}</button>
        </div>
        <p className="help">{t("results_export_types")}</p>
      </div>}
    </div>
  );
}
