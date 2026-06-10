import React from "react";
import { EvalResult, QueryTimings } from "../api/client";
import { BarChart2, CheckCircle, FileText, Zap } from "lucide-react";

interface Props {
  eval: EvalResult;
  backend: string;
  responseTimeMs: number;
  timings?: QueryTimings;
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "text-emerald-400";
  if (score >= 0.4) return "text-amber-400";
  return "text-rose-400";
}

function scoreBg(score: number): string {
  if (score >= 0.7) return "bg-emerald-400/10 border-emerald-400/30";
  if (score >= 0.4) return "bg-amber-400/10 border-amber-400/30";
  return "bg-rose-400/10 border-rose-400/30";
}

function ScorePill({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) {
  return (
    <div
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${scoreBg(value)}`}
    >
      <span className={scoreColor(value)}>{icon}</span>
      <span className="text-slate-300">{label}</span>
      <span className={`font-mono font-semibold ${scoreColor(value)}`}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export default function EvalBadge({ eval: e, backend, responseTimeMs, timings }: Props) {
  return (
    <div className="mt-3 space-y-2">
      <div className="flex flex-wrap gap-2 items-center">
        <ScorePill
          label="Relevance"
          value={e.context_relevance}
          icon={<BarChart2 size={11} />}
        />
        <ScorePill
          label="Grounded"
          value={e.answer_groundedness}
          icon={<CheckCircle size={11} />}
        />
        <ScorePill
          label="Cited"
          value={e.citation_presence}
          icon={<FileText size={11} />}
        />
        <ScorePill
          label="Overall"
          value={e.overall}
          icon={<Zap size={11} />}
        />

        <span className="ml-auto text-xs text-slate-500 flex items-center gap-1 font-mono">
          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
            {backend}
          </span>
          {responseTimeMs.toFixed(0)} ms
        </span>
      </div>

      {/* Per-phase timing breakdown */}
      {timings && (
        <div className="grid grid-cols-4 gap-1 text-xs font-mono text-slate-600">
          <span title="Query embedding time">embed {timings.embedding_ms.toFixed(0)}ms</span>
          <span title="FAISS retrieval time">retrieve {timings.retrieval_ms.toFixed(0)}ms</span>
          <span title="Generation time">gen {timings.generation_ms.toFixed(0)}ms</span>
          <span title="Evaluation time">eval {timings.evaluation_ms.toFixed(0)}ms</span>
        </div>
      )}
    </div>
  );
}
