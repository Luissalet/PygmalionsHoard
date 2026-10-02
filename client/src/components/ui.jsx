import React, { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context.js";
import { KIND_ICON } from "../meta.js";
import { clock, rel } from "../format.js";

export function Icon({ d, size = 17, color }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

export const ICONS = {
  plus: "M12 5v14M5 12h14",
  refresh: "M3 12a9 9 0 0115-6.7L21 8M21 3v5h-5M21 12a9 9 0 01-15 6.7L3 16M3 21v-5h5",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  copy: "M9 9h11v11H9zM5 15V4h11",
  pencil: "M4 20h4L19 9l-4-4L4 16zM13 7l4 4",
  box: "M4 7l8-4 8 4v10l-8 4-8-4zM4 7l8 4 8-4M12 11v10",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  check: "M5 12l5 5 9-10",
  x: "M6 6l12 12M18 6L6 18",
  chevron: "M9 6l6 6-6 6",
  back: "M15 6l-6 6 6 6",
  download: "M12 4v12M7 11l5 5 5-5M4 20h16",
  upload: "M12 16V4M7 9l5-5 5 5M4 16v4h16v-4",
  play: "M7 4l13 8-13 8z",
  stop: "M6 6h12v12H6z",
  clock: "M12 21a9 9 0 100-18 9 9 0 000 18zM12 7v5l3 2",
  search: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5",
  folder: "M3 6h6l2 2h10v11H3z",
  chip: "M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17l9 5 9-5",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  merge: "M6 4v5a6 6 0 006 6 6 6 0 006-6V4M12 15v6",
  compress: "M4 8h16M4 16h16M12 3v5M12 21v-5M9 6l3 2 3-2M9 18l3-2 3 2",
  ruler: "M3 17L17 3l4 4L7 21zM8 12l2 2M11 9l2 2M14 6l2 2",
  tree: "M12 3v6M12 9H5v6M12 9h7v6M5 15v6M19 15v6",
  gear: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12l2-1-1-3-2 .3-1.4-1.4.3-2-3-1-1 2h-2l-1-2-3 1 .3 2L6.8 7.3 5 7 4 10l2 1v2l-2 1 1 3 2-.3 1.4 1.4-.3 2 3 1 1-2h2l1 2 3-1-.3-2 1.4-1.4 2 .3 1-3-2-1z",
  home: "M4 11l8-7 8 7v9h-5v-6H9v6H4z",
  pin: "M9 3h6l-1 6 3 3H7l3-3zM12 12v9",
  alert: "M12 3l10 18H2zM12 10v5M12 18v.5",
};

export function Spinner() {
  return <span className="spinner" role="status" aria-label="…" />;
}

export function Empty({ children }) {
  return <div className="panel help text-center">{children}</div>;
}

export function Field({ label, hint, children, className = "" }) {
  return (
    <label className={`block ${className}`}>
      <span className="label">{label}</span>
      {children}
      {hint && <span className="help mt-1 block">{hint}</span>}
    </label>
  );
}

export function Switch({ checked, onChange, disabled, label }) {
  return (
    <label className="switch" title={label}>
      <input type="checkbox" role="switch" checked={!!checked} disabled={disabled} aria-label={label} onChange={(e) => onChange(e.target.checked)} />
      <span />
    </label>
  );
}

export function Check({ checked, onChange, children, disabled, title }) {
  return (
    <label className="check" title={title}>
      <input type="checkbox" checked={!!checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}

export function Seg({ value, options, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Tabs({ value, options, onChange }) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 border-b" style={{ borderColor: "var(--line)" }}>
      {options.map((o) => (
        <button key={o.value} type="button" role="tab" className="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Chip({ children, className = "", title }) {
  return <span className={`chip ${className}`} title={title}>{children}</span>;
}

export function Section({ title, count, actions, children, id }) {
  return (
    <section className="space-y-2" aria-labelledby={id}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 id={id}>{title}</h2>
        {count !== undefined && count !== null && <span className="chip">{count}</span>}
        <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>
      </div>
      {children}
    </section>
  );
}

export function PageHead({ title, subtitle, actions }) {
  return (
    <header className="mb-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0 flex-1 basis-72">
        <h1>{title}</h1>
        {subtitle && <p className="help mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Busy({ busy, children, ...props }) {
  return (
    <button type="button" {...props} disabled={busy || props.disabled}>
      {busy && <span className="spinner" aria-hidden="true" />}
      {children}
    </button>
  );
}

export function ErrorBox({ error }) {
  const { t } = useApp();
  if (!error) return null;
  return (
    <div className="banner banner-danger" role="alert">
      {t.error(error)}
      {t.hint(error) ? <span className="help block">{t("hint")}: {t.hint(error)}</span> : null}
    </div>
  );
}

export function Rel({ ts }) {
  const { lang } = useApp();
  if (!ts) return <span className="help">—</span>;
  return <time dateTime={new Date(ts * 1000).toISOString()} title={clock(ts, lang)}>{rel(ts, lang)}</time>;
}

export function KindChip({ kind }) {
  const { t } = useApp();
  return (
    <span className="chip">
      <Icon d={KIND_ICON[kind] || KIND_ICON.base} size={12} />
      {t(`kind_${kind}`)}
    </span>
  );
}

export function Bar({ pct, tone = "", thin = false, label }) {
  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  return (
    <div className={`bar ${tone ? `bar-${tone}` : ""} ${thin ? "bar-thin" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value)} aria-label={label}>
      <i style={{ width: `${value}%` }} />
    </div>
  );
}

export function KV({ items }) {
  const rows = items.filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false);
  return (
    <dl className="kv m-0">
      {rows.map(([k, v]) => (
        <React.Fragment key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

export function CopyButton({ text, label }) {
  const { t, notify } = useApp();
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      notify(t("copied"));
    } catch {
      notify(t("copy_failed"), "error");
    }
  };
  return <button type="button" className="btn btn-sm" onClick={copy}><Icon d={ICONS.copy} size={13} />{label || t("copy")}</button>;
}

// ------------------------------------------------------------------ dialogs
export function Modal({ title, onClose, children, wide }) {
  const ref = useRef(null);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    const first = ref.current?.querySelector("input, select, textarea, button");
    first?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal space-y-3" ref={ref} role="dialog" aria-modal="true" aria-label={title} style={wide ? { width: "min(640px, 100%)", maxHeight: "92dvh", overflowY: "auto" } : undefined} onMouseDown={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

export function Drawer({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <aside className="drawer space-y-3" role="dialog" aria-label={title}>
      <div className="flex items-center gap-2">
        <h2 className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{title}</h2>
        <button type="button" className="btn btn-sm" onClick={onClose} aria-label="×"><Icon d={ICONS.x} size={14} /></button>
      </div>
      {children}
    </aside>
  );
}

export function ConfirmDialog({ request, onClose }) {
  const { t } = useApp();
  const cancelRef = useRef(null);
  useEffect(() => {
    if (!request) return undefined;
    cancelRef.current?.focus();
    const onKey = (e) => { if (e.key === "Escape") onClose(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [request, onClose]);
  if (!request) return null;
  return (
    <div className="modal-backdrop" onClick={() => onClose(false)}>
      <div className="modal space-y-3" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="confirm-title">{request.title || t("confirm")}</h2>
        <p style={{ overflowWrap: "anywhere" }}>{request.message}</p>
        <div className="flex justify-end gap-2">
          <button type="button" ref={cancelRef} className="btn" onClick={() => onClose(false)}>{t("cancel")}</button>
          <button type="button" className={`btn ${request.danger === false ? "btn-primary" : "btn-danger"}`} onClick={() => onClose(true)}>{request.confirmLabel || t("delete")}</button>
        </div>
      </div>
    </div>
  );
}

// Loads something on mount and when `deps` change. `reload()` runs it again without flashing the placeholder.
export function useLoad(fn, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const run = useRef(fn);
  run.current = fn;
  const seq = useRef(0);
  const load = useMemo(() => async () => {
    const mine = ++seq.current;
    try {
      const data = await run.current();
      if (mine === seq.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (mine === seq.current) setState((s) => ({ data: s.data, error, loading: false }));
    }
  }, []);
  useEffect(() => {
    setState((s) => ({ ...s, loading: true }));
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { ...state, reload: load };
}

// Busy flags keyed by name; a failing action shows the backend's error and hint in a toast.
export function useBusy() {
  const { toastError } = useApp();
  const [busy, setBusy] = useState({});
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const run = async (key, fn) => {
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      return await fn();
    } catch (error) {
      toastError(error);
      return undefined;
    } finally {
      if (mounted.current) setBusy((b) => ({ ...b, [key]: false }));
    }
  };
  return [busy, run];
}
