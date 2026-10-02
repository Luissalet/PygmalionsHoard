import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorText } from "./api.js";
import { initialLang, makeT, saveLang } from "./i18n.js";
import { AppContext } from "./context.js";
import { ConfirmDialog, Icon, ICONS } from "./components/ui.jsx";
import Panel from "./pages/Panel.jsx";
import Modelos from "./pages/Modelos.jsx";
import Datasets from "./pages/Datasets.jsx";
import Entrenar from "./pages/Entrenar.jsx";
import Trabajos from "./pages/Trabajos.jsx";
import Fusionar from "./pages/Fusionar.jsx";
import Cuantizar from "./pages/Cuantizar.jsx";
import Contexto from "./pages/Contexto.jsx";
import Linaje from "./pages/Linaje.jsx";
import Ajustes from "./pages/Ajustes.jsx";

export { useApp } from "./context.js";

const PAGES = [
  { path: "", key: "nav_panel", icon: ICONS.home, component: Panel },
  { path: "modelos", key: "nav_bases", icon: ICONS.box, component: Modelos },
  { path: "datasets", key: "nav_datasets", icon: ICONS.data, component: Datasets },
  { path: "entrenar", key: "nav_train", icon: ICONS.bolt, component: Entrenar },
  { path: "trabajos", key: "nav_jobs", icon: ICONS.clock, component: Trabajos, badge: "jobs" },
  { path: "fusionar", key: "nav_merge", icon: ICONS.merge, component: Fusionar },
  { path: "cuantizar", key: "nav_quantize", icon: ICONS.compress, component: Cuantizar },
  { path: "contexto", key: "nav_context", icon: ICONS.ruler, component: Contexto },
  { path: "linaje", key: "nav_lineage", icon: ICONS.tree, component: Linaje },
  { path: "ajustes", key: "nav_settings", icon: ICONS.gear, component: Ajustes },
];

function useHashRoute() {
  const read = () => {
    const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
    const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
    return { page: parts[0] || "", param: parts[1] || null, query: new URLSearchParams(query) };
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => { setRoute(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function Toast({ toast, onClose }) {
  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(onClose, toast.kind === "error" ? 9000 : 4500);
    return () => clearTimeout(timer);
  }, [toast, onClose]);
  if (!toast) return null;
  return (
    <div className={`toast ${toast.kind === "error" ? "toast-error" : "toast-ok"}`} role={toast.kind === "error" ? "alert" : "status"} onClick={onClose}>
      {toast.message}
    </div>
  );
}

function SchedulerLight({ scheduler, t }) {
  if (!scheduler) return null;
  const lanes = Object.values(scheduler.lanes || {});
  const busy = lanes.some((l) => l.current);
  const queue = lanes.reduce((n, l) => n + (l.queued || 0), 0);
  const state = !scheduler.enabled || !scheduler.running ? "off" : scheduler.paused ? "paused" : busy ? "busy" : "idle";
  const color = { off: "var(--muted)", paused: "var(--warn)", busy: "var(--accent)", idle: "var(--ok)" }[state];
  return (
    <div className="space-y-0.5" aria-live="polite">
      <div className="flex items-center gap-2 font-semibold text-[12px]" style={{ color: "var(--ink)" }}>
        <span className="dot" style={{ background: color }} />
        {t(`sched_${state}`)}
      </div>
      <div className="help">{t("sched_queue", { n: queue })}</div>
    </div>
  );
}

export default function App() {
  const route = useHashRoute();
  const [lang, setLang] = useState(initialLang);
  const t = useMemo(() => makeT(lang), [lang]);
  const [health, setHealth] = useState(null);
  const [dash, setDash] = useState(null);
  const [dashError, setDashError] = useState(null);
  const [toast, setToast] = useState(null);
  const [confirmReq, setConfirmReq] = useState(null);
  const [version, setVersion] = useState(0);
  const confirmResolve = useRef(null);
  const active = (dash?.counts?.jobs_active || 0) > 0;

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => {
    const n = dash?.counts?.jobs_active || 0;
    document.title = n ? `(${n}) Pygmalion's Hoard` : "Pygmalion's Hoard";
  }, [dash]);

  const lastCounts = useRef(null);
  const refreshDash = useCallback(async () => {
    try {
      const next = await api.dashboard();
      // A count moved (a download finished, a job ended, an artifact appeared): pages that listen to `version` reload too.
      // (a finished job counts too: a pipeline moves from one step to the next without the number of active jobs changing)
      const signature = JSON.stringify([next.counts || {}, next.scheduler?.finished || 0]);
      if (lastCounts.current !== null && lastCounts.current !== signature) setVersion((v) => v + 1);
      lastCounts.current = signature;
      setDash(next);
      setDashError(null);
    } catch (e) {
      setDashError(e);
    }
  }, []);
  // Something changed (a job, an artifact, a setting): reload the dashboard and every page that listens to `version`.
  const changed = useCallback(() => { setVersion((v) => v + 1); refreshDash(); }, [refreshDash]);

  // Every 3 s while jobs are active, every 30 s otherwise, only while the tab is visible; also when it becomes visible again.
  useEffect(() => {
    refreshDash();
    api.health().then(setHealth).catch(() => {});
    const timer = setInterval(() => { if (!document.hidden) refreshDash(); }, active ? 3000 : 30000);
    const onVisible = () => { if (!document.hidden) refreshDash(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => { clearInterval(timer); document.removeEventListener("visibilitychange", onVisible); };
  }, [refreshDash, active]);

  const notify = useCallback((message, kind = "ok") => setToast({ message, kind, id: Math.random() }), []);
  const toastError = useCallback((error) => setToast({ message: errorText(error, t), kind: "error", id: Math.random() }), [t]);
  const confirm = useCallback((request) => new Promise((resolve) => {
    confirmResolve.current = resolve;
    setConfirmReq(request);
  }), []);
  const closeConfirm = useCallback((answer) => {
    setConfirmReq(null);
    if (confirmResolve.current) confirmResolve.current(answer);
    confirmResolve.current = null;
  }, []);

  const changeLang = useCallback((next) => {
    saveLang(next);
    setLang(next);
    api.call("settings_set", { values: { "ui.language": next } }).catch(() => {});
  }, []);

  const value = useMemo(() => ({
    t, lang, setLang: changeLang, health, dash, dashError, refreshDash, changed, version, notify, toastError, confirm, route,
  }), [t, lang, changeLang, health, dash, dashError, refreshDash, changed, version, notify, toastError, confirm, route]);

  const page = PAGES.find((p) => p.path === route.page) || PAGES[0];
  const Component = page.component;
  // The badge counts what is running or waiting, nothing else: failed jobs stay in the list and on the panel.
  const badges = { jobs: dash?.counts?.jobs_active || 0 };

  return (
    <AppContext.Provider value={value}>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 skip-link focus:p-2">{t("skip")}</a>
      <div className="min-h-dvh md:grid md:grid-cols-[220px_minmax(0,1fr)]">
        <aside className="sticky top-0 z-10 border-b md:flex md:h-dvh md:flex-col md:self-start md:overflow-y-auto md:border-b-0 md:border-r" style={{ background: "var(--sidebar)", borderColor: "var(--line)" }}>
          <div className="flex items-center gap-3 px-4 py-3 md:py-4">
            <img src="/icon-192.png" alt="" width="32" height="32" className="brand-icon rounded-lg" decoding="sync" />
            <div>
              <div>Pygmalion's Hoard</div>
              <div className="brand-sub">{t("subtitle")}</div>
            </div>
          </div>
          <nav aria-label={t("sections")} className="flex gap-1 overflow-x-auto px-3 py-2 md:flex-col">
            {PAGES.map((p) => {
              const n = p.badge ? badges[p.badge] : 0;
              return (
                <a key={p.path} href={`#/${p.path}`} className="nav-link shrink-0 text-[13px]" aria-current={p.path === page.path ? "page" : undefined}>
                  <Icon d={p.icon} />
                  {t(p.key)}
                  {n > 0 && <span className="nav-badge" aria-label={t("badge_n", { n })}>{n > 99 ? "99+" : n}</span>}
                </a>
              );
            })}
          </nav>
          <div className="hidden flex-1 md:block" />
          <div className="hidden space-y-3 border-t px-4 py-3 md:block" style={{ borderColor: "var(--line)" }}>
            <SchedulerLight scheduler={dash?.scheduler} t={t} />
            {health?.offline && <span className="chip chip-amber">{t("offline_on")}</span>}
            <button type="button" className="btn btn-sm" onClick={() => changeLang(lang === "es" ? "en" : "es")}>{t("language")}</button>
          </div>
        </aside>
        <main id="main" className="min-w-0 px-4 py-4 md:px-7 md:py-6">
          {dashError && !dash && (
            <div className="banner banner-danger mb-4" role="alert">
              {t("unreachable")}: {t.error(dashError)}. <button type="button" className="btn-link" onClick={refreshDash}>{t("retry")}</button>
            </div>
          )}
          {dashError && dash && (
            <div className="banner banner-warn mb-4" role="status">
              {t("stale")}: {t.error(dashError)}. <button type="button" className="btn-link" onClick={refreshDash}>{t("retry")}</button>
            </div>
          )}
          <Component key={`${route.page}/${route.param || ""}`} param={route.param} query={route.query} />
          <div className="mt-8 flex flex-wrap items-center gap-3 border-t pt-3 md:hidden" style={{ borderColor: "var(--line)" }}>
            <SchedulerLight scheduler={dash?.scheduler} t={t} />
            <button type="button" className="btn btn-sm" onClick={() => changeLang(lang === "es" ? "en" : "es")}>{t("language")}</button>
          </div>
        </main>
      </div>
      <Toast toast={toast} onClose={() => setToast(null)} />
      <ConfirmDialog request={confirmReq} onClose={closeConfirm} />
    </AppContext.Provider>
  );
}
