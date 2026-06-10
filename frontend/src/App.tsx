import React, { useCallback, useEffect, useRef, useState } from "react";
import { clearCache, fetchHealth, fetchMetrics, HealthResponse, MetricsResponse } from "./api/client";
import ChatPanel from "./components/ChatPanel";
import ConversationSidebar from "./components/ConversationSidebar";
import DocumentUpload from "./components/DocumentUpload";
import MetricsChart from "./components/MetricsChart";
import { Activity, BookOpen, Github, Menu, Server, Trash2, X } from "lucide-react";

function generateId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [indexSize, setIndexSize] = useState(0);
  const [hasDocuments, setHasDocuments] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [conversationId, setConversationId] = useState<string>(() => generateId());
  const chatResetRef = useRef<(() => void) | null>(null);

  const pollHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
      setIndexSize(h.index_size);
      setHasDocuments(h.index_size > 0);
      setHealthError(false);
    } catch {
      setHealthError(true);
    }
  }, []);

  const pollMetrics = useCallback(async () => {
    try {
      const m = await fetchMetrics();
      setMetrics(m);
    } catch {
      // Metrics non-critical — fail silently
    }
  }, []);

  useEffect(() => {
    pollHealth();
    const id = setInterval(pollHealth, 15_000);
    return () => clearInterval(id);
  }, [pollHealth]);

  useEffect(() => {
    pollMetrics();
    const id = setInterval(pollMetrics, 30_000);
    return () => clearInterval(id);
  }, [pollMetrics]);

  const onIngested = useCallback(
    (_doc: { filename: string; chunks: number }) => {
      setHasDocuments(true);
      pollHealth();
      pollMetrics();
    },
    [pollHealth, pollMetrics]
  );

  const startNewConversation = useCallback(() => {
    const newId = generateId();
    setConversationId(newId);
    chatResetRef.current?.();
  }, []);

  const handleCacheClean = useCallback(async () => {
    try {
      await clearCache();
      pollMetrics();
    } catch {
      // non-critical
    }
  }, [pollMetrics]);
  return (
    <div className="flex flex-col h-screen overflow-hidden relative bg-slate-950 font-sans text-slate-100 antialiased">
      {/* Background Mesh and Orbs */}
      <div className="bg-mesh" />
      <div className="glowing-orb-1" />
      <div className="glowing-orb-2" />

      {/* Top nav */}
      <header className="shrink-0 flex items-center justify-between px-4 h-14 border-b border-white/5 bg-slate-900/40 backdrop-blur-md z-10 shadow-lg">
        <div className="flex items-center gap-2.5">
          {/* Mobile sidebar toggle */}
          <button
            className="sm:hidden text-slate-400 hover:text-slate-200 transition-colors mr-1"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-600 to-teal-400 flex items-center justify-center shadow-md shadow-brand-500/20">
            <BookOpen size={15} className="text-white" />
          </div>
          <span className="font-semibold text-slate-100 text-sm tracking-tight text-glow-gradient">
            Enterprise RAG Copilot
          </span>
          {health && (
            <span className="hidden sm:inline text-[10px] text-slate-500 font-mono bg-slate-900/60 px-1.5 py-0.5 rounded border border-white/5">
              v{health.version}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Backend status */}
          <div className="flex items-center gap-1.5 text-xs bg-slate-900/40 px-2.5 py-1 rounded-full border border-white/5">
            <div
              className={`w-1.5 h-1.5 rounded-full animate-pulse ${
                healthError ? "bg-rose-500" : "bg-emerald-500"
              }`}
            />
            <span className="text-slate-400 font-medium">
              {healthError ? "API offline" : "API online"}
            </span>
          </div>

          {health && (
            <div className="hidden sm:flex items-center gap-1 text-xs text-slate-400 bg-slate-900/40 px-2.5 py-1 rounded-full border border-white/5">
              <Server size={11} className="text-slate-500" />
              <span className="font-mono truncate max-w-[140px]">
                {health.embed_model.split("/").pop()}
              </span>
            </div>
          )}

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-slate-300 transition-colors p-1.5 rounded-lg hover:bg-white/5"
            aria-label="GitHub"
          >
            <Github size={16} />
          </a>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden relative z-10">
        {/* Sidebar — hidden on mobile unless toggled */}
        <aside
          className={`${
            sidebarOpen ? "flex" : "hidden"
          } sm:flex flex-col sm:w-72 w-full shrink-0 border-r border-white/5 bg-slate-950/45 backdrop-blur-md overflow-y-auto`}
        >
          <div className="p-4 border-b border-white/5">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Document Ingestion
            </h2>
            <DocumentUpload onIngested={onIngested} indexSize={indexSize} />
          </div>

          {/* System Status panel */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <Activity size={11} />
              System Status
            </h2>
            <div className="flex flex-col gap-2.5 text-xs text-slate-400">
              <div className="flex justify-between items-center">
                <span>API</span>
                <span className={`font-mono px-2 py-0.5 rounded text-[10px] font-semibold ${
                  healthError ? "bg-rose-500/10 text-rose-400 border border-rose-500/20" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                }`}>
                  {healthError ? "offline" : "online"}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span>Index size</span>
                <span className="font-mono text-slate-300">{indexSize} vectors</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Embed model</span>
                <span className="font-mono text-slate-300 truncate ml-2 max-w-[120px] text-right" title={health?.embed_model ?? ""}>
                  {health?.embed_model.split("/").pop() ?? "—"}
                </span>
              </div>
              {metrics && (
                <>
                  <div className="flex justify-between items-center">
                    <span>Queries served</span>
                    <span className="font-mono text-slate-300">{metrics.total_queries}</span>
                  </div>
                  {metrics.total_queries > 0 && (
                    <>
                      <div className="flex justify-between items-center">
                        <span>Avg latency</span>
                        <span className="font-mono text-slate-300">
                          {metrics.avg_response_time_ms.toFixed(0)} ms
                        </span>
                      </div>
                      {metrics.latency_percentiles && (
                        <div className="flex justify-between items-center">
                          <span>P95 latency</span>
                          <span className="font-mono text-slate-300">
                            {metrics.latency_percentiles.p95_ms.toFixed(0)} ms
                          </span>
                        </div>
                      )}
                    </>
                  )}
                  <div className="flex justify-between items-center">
                    <span>Gen backend</span>
                    <span className="font-mono text-slate-300">
                      {metrics.generator_backend_status.active_mode}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>OpenAI</span>
                    <span className={`font-mono px-2 py-0.5 rounded text-[10px] font-semibold ${
                      metrics.generator_backend_status.openai === "configured"
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-slate-800/40 text-slate-500"
                    }`}>
                      {metrics.generator_backend_status.openai}
                    </span>
                  </div>
                  {metrics.cache && (
                    <div className="flex justify-between items-center">
                      <span>Cache hit rate</span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-slate-300">
                          {(metrics.cache.hit_rate * 100).toFixed(0)}%
                          {" "}({metrics.cache.size})
                        </span>
                        <button
                          onClick={handleCacheClean}
                          title="Clear cache"
                          className="text-slate-500 hover:text-rose-400 transition-colors p-0.5 hover:bg-white/5 rounded"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Metrics chart */}
          <div className="p-4 border-b border-white/5">
            <MetricsChart />
          </div>

          {/* Conversation history */}
          <div className="p-4 border-b border-white/5">
            <ConversationSidebar
              currentConversationId={conversationId}
              onSelect={(id) => {
                setConversationId(id);
                chatResetRef.current?.();
              }}
              onNew={startNewConversation}
            />
          </div>

          {/* Footer info */}
          <div className="p-4 mt-auto border-t border-white/5">
            <div className="flex flex-col gap-2 text-[10px] text-slate-500">
              <div className="flex justify-between">
                <span>Vector store</span>
                <span className="font-mono text-slate-450">FAISS · CPU</span>
              </div>
              <div className="flex justify-between">
                <span>Backend</span>
                <span className="font-mono text-slate-450">FastAPI v3</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Chat */}
        <main className="flex-1 overflow-hidden">
          <ChatPanel
            hasDocuments={hasDocuments}
            conversationId={conversationId}
            onResetRef={chatResetRef}
          />
        </main>
      </div>
    </div>
  );
}
