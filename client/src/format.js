// Numbers, sizes, times. es-ES by default, en-GB when English is chosen.
export const locale = (lang) => (lang === "en" ? "en-GB" : "es-ES");

export function num(value, digits = 0, lang = "es") {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(locale(lang), { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// A difference with its sign: +0,12 / -0,03.
export function signed(value, digits = 2, lang = "es") {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${num(Math.abs(v), digits, lang)}`;
}

// 1536 -> "1,5 GB" from megabytes.
export function mb(value, lang = "es") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  return v >= 1024 ? `${num(v / 1024, 1, lang)} GB` : `${num(v, 0, lang)} MB`;
}

export function bytes(value, lang = "es") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  if (v < 1024) return `${num(v, 0, lang)} B`;
  if (v < 1024 ** 2) return `${num(v / 1024, 0, lang)} KB`;
  if (v < 1024 ** 3) return `${num(v / 1024 ** 2, 1, lang)} MB`;
  return `${num(v / 1024 ** 3, 2, lang)} GB`;
}

export function compact(value, lang = "es") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  if (v >= 1e9) return `${num(v / 1e9, 1, lang)} G`;
  if (v >= 1e6) return `${num(v / 1e6, 1, lang)} M`;
  if (v >= 1e3) return `${num(v / 1e3, 1, lang)} k`;
  return num(v, 0, lang);
}

// Seconds -> "2 h 5 min", "4 min", "35 s".
export function seconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const s = Math.round(Number(value));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

export function clock(ts, lang) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(locale(lang), { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).replace(",", "");
}

export function shortClock(ts, lang) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(locale(lang), { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// "hace 5 min" / "dentro de 2 h"
export function rel(ts, lang, nowMs = Date.now()) {
  if (!ts) return "—";
  const diff = ts * 1000 - nowMs;
  const abs = Math.abs(diff);
  const formatter = new Intl.RelativeTimeFormat(locale(lang), { numeric: "auto", style: "short" });
  if (abs < 45_000) return formatter.format(0, "second");
  const units = [["day", 86_400_000], ["hour", 3_600_000], ["minute", 60_000]];
  for (const [unit, size] of units) {
    if (abs >= size || unit === "minute") return formatter.format(Math.round(diff / size), unit);
  }
  return "—";
}

export const splitList = (text) => (text || "").split(/[\n,;]+/).map((s) => s.trim()).filter(Boolean);

export const toNumber = (text) => {
  if (text === "" || text === null || text === undefined) return undefined;
  const n = Number(String(text).replace(",", "."));
  return Number.isFinite(n) ? n : undefined;
};

// Drops empty values so the backend keeps its defaults.
export const clean = (obj) => Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== "" && v !== undefined && v !== null && !(Array.isArray(v) && !v.length)));
