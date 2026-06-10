import React, { useEffect, useState } from "react";
import { fetchMetricsTimeseries, MetricsTimeseriesPoint } from "../api/client";
import { BarChart2, TrendingUp } from "lucide-react";

export default function MetricsChart() {
  const [points, setPoints] = useState<MetricsTimeseriesPoint[]>([]);
  const [tab, setTab] = useState<"queries" | "latency" | "groundedness">("queries");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  useEffect(() => {
    fetchMetricsTimeseries(24, 60)
      .then((r) => setPoints(r.points))
      .catch(() => {});
    const id = setInterval(() => {
      fetchMetricsTimeseries(24, 60)
        .then((r) => setPoints(r.points))
        .catch(() => {});
    }, 60_000);
    return () => clearInterval(id);
  }, []);

  if (points.length === 0) {
    return (
      <div className="text-xs text-slate-600 text-center py-4 bg-slate-900/10 rounded-lg border border-white/5 font-mono">
        No telemetry data available
      </div>
    );
  }

  const getValue = (p: MetricsTimeseriesPoint) => {
    if (tab === "queries") return p.queries;
    if (tab === "latency") return p.avg_latency_ms;
    return p.avg_groundedness * 100;
  };

  const maxVal = Math.max(...points.map(getValue), 1);

  const formatTs = (ts: number) => {
    const d = new Date(ts * 1000);
    return `${d.getHours().toString().padStart(2, "0")}:00`;
  };

  const tabs: { key: typeof tab; label: string }[] = [
    { key: "queries", label: "Queries" },
    { key: "latency", label: "Latency" },
    { key: "groundedness", label: "Groundedness" },
  ];

  // SVG Chart Config
  const width = 280;
  const height = 70;
  const paddingX = 0;
  const paddingY = 8;

  const pointsCoords = points.map((p, i) => {
    const val = getValue(p);
    const x = points.length > 1 ? paddingX + (i / (points.length - 1)) * (width - paddingX * 2) : paddingX;
    const y = height - paddingY - (maxVal > 0 ? (val / maxVal) * (height - paddingY * 2) : 0);
    return { x, y, val, timestamp: p.timestamp };
  });

  const linePath = pointsCoords.length > 1
    ? "M " + pointsCoords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" L ")
    : "";

  const areaPath = pointsCoords.length > 1
    ? `${linePath} L ${width.toFixed(1)},${height.toFixed(1)} L 0,${height.toFixed(1)} Z`
    : "";

  const currentHoveredPoint = hoveredIdx !== null ? pointsCoords[hoveredIdx] : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <TrendingUp size={11} className="text-brand-400" />
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
            24h Metrics
          </span>
        </div>

        {currentHoveredPoint ? (
          <span className="text-[10px] font-mono text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded border border-teal-500/20">
            {formatTs(currentHoveredPoint.timestamp)}:{" "}
            <span className="font-semibold">
              {tab === "latency"
                ? `${currentHoveredPoint.val.toFixed(0)}ms`
                : tab === "groundedness"
                ? `${currentHoveredPoint.val.toFixed(0)}%`
                : currentHoveredPoint.val}
            </span>
          </span>
        ) : (
          <span className="text-[10px] font-mono text-slate-650">
            Hover graph to view
          </span>
        )}
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 bg-slate-950 p-0.5 rounded-lg border border-white/5">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => {
              setTab(key);
              setHoveredIdx(null);
            }}
            className={`flex-1 text-[10px] font-medium py-1 px-1.5 rounded-md transition-all duration-300 ${
              tab === key
                ? "bg-slate-900 text-brand-350 shadow-sm border border-white/5"
                : "text-slate-500 hover:text-slate-350"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Chart SVG */}
      <div className="relative h-[70px] mt-2 select-none overflow-hidden rounded-lg bg-slate-900/10 border border-white/5">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full overflow-visible"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.00" />
            </linearGradient>
            <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="50%" stopColor="#818cf8" />
              <stop offset="100%" stopColor="#2dd4bf" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line x1="0" y1={height / 2} x2={width} y2={height / 2} className="stroke-white/[0.03] stroke-1" />
          <line x1="0" y1={height - 2} x2={width} y2={height - 2} className="stroke-white/[0.06] stroke-1" />

          {pointsCoords.length > 1 && (
            <>
              {/* Area */}
              <path d={areaPath} fill="url(#areaGrad)" />

              {/* Line */}
              <path d={linePath} fill="none" stroke="url(#lineGrad)" strokeWidth="1.5" strokeLinecap="round" />
            </>
          )}

          {/* Hover Crosshairs & Tracker Dots */}
          {currentHoveredPoint && (
            <>
              <line
                x1={currentHoveredPoint.x}
                y1={0}
                x2={currentHoveredPoint.x}
                y2={height}
                stroke="#6366f1"
                strokeOpacity="0.3"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
              <circle
                cx={currentHoveredPoint.x}
                cy={currentHoveredPoint.y}
                r="5"
                fill="#6366f1"
                fillOpacity="0.3"
                className="animate-ping"
              />
              <circle
                cx={currentHoveredPoint.x}
                cy={currentHoveredPoint.y}
                r="3.5"
                fill="#2dd4bf"
                stroke="#030712"
                strokeWidth="1.5"
              />
            </>
          )}
        </svg>

        {/* Hover detection layers */}
        <div className="absolute inset-0 flex z-20">
          {points.map((_, idx) => (
            <div
              key={idx}
              className="flex-1 h-full cursor-crosshair"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            />
          ))}
        </div>
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between px-0.5 text-[9px] text-slate-700 font-mono">
        <span>{formatTs(points[0].timestamp)}</span>
        <span>{formatTs(points[points.length - 1].timestamp)}</span>
      </div>
    </div>
  );
}
