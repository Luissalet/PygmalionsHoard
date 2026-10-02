import React from "react";
import { num } from "../format.js";

const niceTicks = (min, max, count = 4) => {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (max === min) return [min];
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + step * i);
};

// Loss curves. series: [{ name, className, points: [[x, y], ...] }]
export function LineChart({ series, height = 190, xLabel, lang = "es", ariaLabel }) {
  const width = 640;
  const pad = { l: 44, r: 12, t: 10, b: 26 };
  const all = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!all.length) return null;
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMax === yMin) { yMax += 0.5; yMin = Math.max(0, yMin - 0.5); }
  const gap = (yMax - yMin) * 0.08;
  yMin = Math.max(0, yMin - gap);
  yMax += gap;
  const sx = (x) => pad.l + ((x - xMin) / (xMax === xMin ? 1 : xMax - xMin)) * (width - pad.l - pad.r);
  const sy = (y) => height - pad.b - ((y - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const path = (points) => points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
  const yTicks = niceTicks(yMin, yMax, 4);
  const xTicks = niceTicks(xMin, xMax, 4);
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={ariaLabel} style={{ maxHeight: height * 1.6 }}>
      {yTicks.map((y) => (
        <g key={`y${y}`}>
          <line className="grid-line" x1={pad.l} x2={width - pad.r} y1={sy(y)} y2={sy(y)} />
          <text className="axis-text" x={pad.l - 6} y={sy(y) + 3} textAnchor="end">{num(y, 2, lang)}</text>
        </g>
      ))}
      {xTicks.map((x) => (
        <text key={`x${x}`} className="axis-text" x={sx(x)} y={height - 8} textAnchor="middle">{num(Math.round(x), 0, lang)}</text>
      ))}
      {xLabel && <text className="axis-text" x={width - pad.r} y={height - 8} textAnchor="end">{xLabel}</text>}
      {series.map((s) => s.points.length > 1
        ? <path key={s.name} className={s.className} d={path(s.points)} />
        : s.points.map((p) => <circle key={`${s.name}${p[0]}`} cx={sx(p[0])} cy={sy(p[1])} r="3.5" fill="var(--hoard-info)" />))}
      {series.filter((s) => s.points.length > 1 && s.dots).map((s) => s.points.map((p) => <circle key={`${s.name}d${p[0]}`} cx={sx(p[0])} cy={sy(p[1])} r="3" fill="var(--hoard-info)" />))}
    </svg>
  );
}

export function Sparkline({ points, width = 120, height = 28, ariaLabel }) {
  const pts = (points || []).filter((p) => Array.isArray(p) && Number.isFinite(p[1]));
  if (pts.length < 2) return <span className="help">—</span>;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const sx = (x) => 1 + ((x - xMin) / (xMax === xMin ? 1 : xMax - xMin)) * (width - 2);
  const sy = (y) => height - 2 - ((y - yMin) / (yMax === yMin ? 1 : yMax - yMin)) * (height - 4);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel} style={{ flex: "none" }}>
      <path d={d} fill="none" stroke="var(--hoard-accent)" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

// Length histogram: buckets [{ bucket, count }]
export function Histogram({ buckets, height = 110, ariaLabel }) {
  const width = 360;
  const list = buckets || [];
  const max = Math.max(1, ...list.map((b) => b.count));
  const w = list.length ? (width - 8) / list.length : 0;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={ariaLabel} style={{ maxHeight: 170 }}>
      {list.map((b, i) => {
        const h = Math.round((b.count / max) * (height - 34));
        return (
          <g key={b.bucket}>
            <rect className="hist-bar" x={4 + i * w + 3} y={height - 22 - h} width={Math.max(2, w - 6)} height={h} rx="2" />
            <text className="axis-text" x={4 + i * w + w / 2} y={height - 8} textAnchor="middle">{b.bucket}</text>
            {b.count > 0 && <text className="axis-text" x={4 + i * w + w / 2} y={height - 25 - h} textAnchor="middle">{b.count}</text>}
          </g>
        );
      })}
    </svg>
  );
}
