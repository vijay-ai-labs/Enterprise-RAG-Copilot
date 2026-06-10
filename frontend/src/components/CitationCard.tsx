import React, { useState } from "react";
import { Citation } from "../api/client";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  citations: Citation[];
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const gradient =
    score >= 0.7
      ? "from-emerald-500 to-teal-400"
      : score >= 0.4
      ? "from-amber-500 to-orange-450"
      : "from-rose-500 to-red-400";
  
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-slate-950 rounded-full overflow-hidden border border-white/5">
        <div className={`h-full bg-gradient-to-r ${gradient} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono text-slate-400 w-7 text-right">{pct}%</span>
    </div>
  );
}

export default function CitationCard({ citations }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (citations.length === 0) return null;

  return (
    <div className="mt-3.5">
      <div className="flex items-center gap-1.5 mb-2 text-[10px] text-slate-500 font-semibold uppercase tracking-wider">
        <BookOpen size={11} className="text-brand-400" />
        <span>Sources ({citations.length})</span>
      </div>

      <div className="flex flex-col gap-2">
        {citations.map((c, i) => {
          const isExpanded = expanded === i;
          return (
            <div
              key={c.chunk_id}
              className={`rounded-xl border transition-all duration-300 overflow-hidden ${
                isExpanded
                  ? "bg-slate-900/60 border-brand-500/30 shadow-md shadow-brand-500/5"
                  : "bg-slate-900/25 border-white/5 hover:border-white/10 hover:bg-slate-900/40"
              }`}
            >
              <button
                className="w-full flex items-center justify-between px-3.5 py-2.5 text-left transition-colors"
                onClick={() => setExpanded(isExpanded ? null : i)}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="shrink-0 w-5 h-5 rounded-md bg-gradient-to-tr from-brand-600 to-indigo-650 flex items-center justify-center text-[10px] font-bold text-white shadow-sm">
                    {i + 1}
                  </span>
                  <span className="text-xs font-semibold text-slate-200 truncate">
                    {c.filename}
                  </span>
                  <span className="shrink-0 text-[10px] text-slate-400 font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-white/5">
                    p. {c.page}
                  </span>
                </div>

                <div className="flex items-center gap-3 shrink-0 ml-3">
                  <div className="w-20 hidden sm:block">
                    <ScoreBar score={c.score} />
                  </div>
                  <div className={`p-1 rounded-md transition-colors ${isExpanded ? "bg-brand-500/10 text-brand-300" : "text-slate-500"}`}>
                    {isExpanded ? (
                      <ChevronUp size={14} />
                    ) : (
                      <ChevronDown size={14} />
                    )}
                  </div>
                </div>
              </button>

              {/* Smooth slide panel */}
              <div
                className={`grid transition-all duration-300 ease-in-out ${
                  isExpanded ? "grid-rows-[1fr] opacity-100 border-t border-white/5" : "grid-rows-[0fr] opacity-0"
                }`}
              >
                <div className="overflow-hidden">
                  <div className="p-3.5 bg-slate-950/40 space-y-2">
                    <p className="text-xs text-slate-350 leading-relaxed font-mono italic">
                      "{c.text_snippet}
                      {c.text_snippet.length >= 200 ? "…" : ""}"
                    </p>
                    <div className="flex items-center justify-between pt-1 text-[10px] text-slate-500 font-mono">
                      <span>Similarity score</span>
                      <span className="text-brand-300 font-semibold">{c.score.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
