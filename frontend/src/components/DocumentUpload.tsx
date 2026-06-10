import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ingestFile,
  fetchDocuments,
  deleteDocument,
  IngestResponse,
  IngestTimings,
  DocumentInfo,
} from "../api/client";
import { CheckCircle2, FileText, Loader2, Trash2, Upload, XCircle, Clock } from "lucide-react";

interface IngestedDoc {
  filename: string;
  chunks: number;
}

interface Props {
  onIngested: (doc: IngestedDoc) => void;
  indexSize: number;
}

type UploadState = "idle" | "uploading" | "success" | "error";

const SUPPORTED_TYPES = ".pdf,.txt,.md,.markdown,.docx,.pptx,.html,.htm";
const SUPPORTED_LABEL = "PDF · TXT · MD · DOCX · PPTX · HTML";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentUpload({ onIngested, indexSize }: Props) {
  const [state, setState] = useState<UploadState>("idle");
  const [message, setMessage] = useState("");
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [dragging, setDragging] = useState(false);
  const [lastTimings, setLastTimings] = useState<IngestTimings | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetchDocuments();
      setDocs(res.documents);
    } catch {
      // Non-critical; fail silently
    }
  }, []);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  const handleFile = useCallback(
    async (file: File) => {
      setState("uploading");
      setMessage("");
      setLastTimings(null);
      try {
        const res: IngestResponse = await ingestFile(file);
        onIngested({ filename: res.filename, chunks: res.chunks_stored });
        setState("success");
        setMessage(`${res.chunks_stored} chunks indexed`);
        if (res.timings) setLastTimings(res.timings);
        await refreshDocs();
      } catch (err: any) {
        setState("error");
        setMessage(err?.response?.data?.detail ?? err?.message ?? "Upload failed");
      }
    },
    [onIngested, refreshDocs]
  );

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Remove "${filename}" from the index?`)) return;
    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      await refreshDocs();
    } catch (err: any) {
      alert(err?.response?.data?.detail ?? "Delete failed");
    } finally {
      setDeletingId(null);
    }
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  const dropZoneCls = [
    "relative flex flex-col items-center justify-center gap-3 p-5 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer text-center overflow-hidden",
    dragging
      ? "border-brand-500 bg-brand-500/10 shadow-lg shadow-brand-500/5 scale-[0.98]"
      : state === "uploading"
      ? "border-slate-700 bg-slate-900/20 pointer-events-none opacity-60"
      : "border-slate-800 hover:border-brand-500/50 hover:bg-slate-900/30 bg-slate-900/10",
  ].join(" ");

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        className={dropZoneCls}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={SUPPORTED_TYPES}
          className="hidden"
          onChange={onInputChange}
        />

        {state === "uploading" ? (
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-brand-500/10 text-brand-400">
            <Loader2 size={20} className="animate-spin" />
          </div>
        ) : (
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-slate-950 text-slate-400 border border-white/5 group-hover:text-brand-400 transition-colors">
            <Upload size={18} />
          </div>
        )}

        <div>
          <p className="text-xs font-semibold text-slate-200">
            {state === "uploading" ? "Analyzing Document…" : "Drop a file or browse files"}
          </p>
          <p className="text-[10px] text-slate-500 mt-1">{SUPPORTED_LABEL}</p>
        </div>

        {state === "success" && (
          <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full mt-1">
            <CheckCircle2 size={11} />
            {message}
          </div>
        )}
        {state === "error" && (
          <div className="flex items-center gap-1.5 text-[10px] text-rose-400 font-semibold bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full mt-1">
            <XCircle size={11} />
            {message}
          </div>
        )}
      </div>

      {/* Ingestion timing breakdown */}
      {state === "success" && lastTimings && (
        <div className="flex flex-col border border-white/5 rounded-xl bg-slate-900/35 overflow-hidden">
          <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-white/5 bg-slate-950 text-[10px] text-slate-500 font-semibold uppercase tracking-wider">
            <Clock size={11} className="text-brand-400" />
            <span>Telemetry Breakdown</span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px] font-mono p-3">
            <span className="text-slate-500">Extraction</span>
            <span className="text-slate-300 text-right">{lastTimings.extraction_ms.toFixed(0)} ms</span>
            <span className="text-slate-500">Chunking</span>
            <span className="text-slate-300 text-right">{lastTimings.chunking_ms.toFixed(0)} ms</span>
            <span className="text-slate-500">Embedding</span>
            <span className="text-slate-300 text-right">{lastTimings.embedding_ms.toFixed(0)} ms</span>
            <span className="text-slate-500">Indexing</span>
            <span className="text-slate-300 text-right">{lastTimings.indexing_ms.toFixed(0)} ms</span>
            <span className="text-brand-300 font-semibold border-t border-white/5 pt-1.5">Total Duration</span>
            <span className="text-brand-400 font-semibold text-right border-t border-white/5 pt-1.5">
              {lastTimings.total_ms.toFixed(0)} ms
            </span>
          </div>
        </div>
      )}

      {/* Document list with delete */}
      {docs.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-1">
            Documents ({docs.length})
          </p>
          <div className="flex flex-col gap-2 max-h-56 overflow-y-auto pr-1">
            {docs.map((doc) => {
              const fileExt = doc.filename.split(".").pop()?.toUpperCase() ?? "DOC";
              return (
                <div
                  key={doc.doc_id}
                  className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-slate-900/35 border border-white/5 group hover:border-white/10 transition-colors"
                >
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-slate-950 border border-white/5 flex items-center justify-center text-[9px] font-extrabold text-brand-400">
                    {fileExt}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-slate-200 truncate" title={doc.filename}>{doc.filename}</p>
                    <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                      {doc.chunks_stored} chunks · {formatBytes(doc.file_size_bytes)}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.doc_id, doc.filename)}
                    disabled={deletingId === doc.doc_id}
                    title="Remove document"
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1.5 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-all disabled:opacity-30"
                  >
                    {deletingId === doc.doc_id ? (
                      <Loader2 size={13} className="animate-spin text-brand-400" />
                    ) : (
                      <Trash2 size={13} />
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
