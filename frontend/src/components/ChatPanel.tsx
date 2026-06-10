import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamQuery, submitFeedback, QueryResponse } from "../api/client";
import CitationCard from "./CitationCard";
import EvalBadge from "./EvalBadge";
import { Bot, Loader2, Plus, Send, ThumbsDown, ThumbsUp, User } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  error?: string;
  feedback?: 1 | -1 | null;
}

interface Props {
  hasDocuments: boolean;
  conversationId: string;
  onResetRef?: React.MutableRefObject<(() => void) | null>;
}

const SAMPLE_QUERIES = [
  "What is Retrieval-Augmented Generation?",
  "How does chunking affect RAG quality?",
  "What are the main branches of machine learning?",
  "Explain the Transformer architecture.",
];

export default function ChatPanel({ hasDocuments, conversationId, onResetRef }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loadingPhase, setLoadingPhase] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Expose reset so parent can clear chat when conversation changes
  useEffect(() => {
    if (onResetRef) {
      onResetRef.current = () => {
        abortRef.current?.abort();
        setMessages([]);
        setInput("");
        setLoadingPhase(null);
        setStreamingContent("");
      };
    }
  }, [onResetRef]);

  // Clear messages when conversation ID changes externally
  const prevConvId = useRef(conversationId);
  useEffect(() => {
    if (prevConvId.current !== conversationId) {
      prevConvId.current = conversationId;
      abortRef.current?.abort();
      setMessages([]);
      setInput("");
      setLoadingPhase(null);
      setStreamingContent("");
    }
  }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingPhase]);

  const submit = (query: string) => {
    if (!query.trim() || loadingPhase !== null) return;

    setMessages((prev) => [...prev, { role: "user", content: query.trim() }]);
    setInput("");
    setLoadingPhase("Retrieving sources…");

    abortRef.current = streamQuery(
      query.trim(),
      undefined,
      (event) => {
        if (event.event === "retrieving") {
          setLoadingPhase("Retrieving sources…");
        } else if (event.event === "generating") {
          setLoadingPhase("generating");
          setStreamingContent("");
        } else if (event.event === "token") {
          // Accumulate tokens into live streaming bubble
          setStreamingContent((prev) => prev + event.token);
          setLoadingPhase("generating");
        } else if (event.event === "evaluating") {
          setLoadingPhase("Evaluating quality…");
        } else if (event.event === "done") {
          setStreamingContent("");
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: event.data.answer,
              response: event.data,
              feedback: null,
            },
          ]);
          setLoadingPhase(null);
        } else if (event.event === "error") {
          setStreamingContent("");
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: "An error occurred.", error: event.message },
          ]);
          setLoadingPhase(null);
        }
      },
      () => {
        setStreamingContent("");
        setLoadingPhase(null);
        inputRef.current?.focus();
      },
      (err) => {
        setStreamingContent("");
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "An error occurred.", error: err.message },
        ]);
        setLoadingPhase(null);
        inputRef.current?.focus();
      },
      conversationId,
    );
  };

  const handleFeedback = async (msgIdx: number, rating: 1 | -1) => {
    const msg = messages[msgIdx];
    if (!msg.response?.query_id || msg.feedback != null) return;

    // Optimistic update
    setMessages((prev) =>
      prev.map((m, i) => (i === msgIdx ? { ...m, feedback: rating } : m))
    );

    try {
      await submitFeedback({
        query_id: msg.response.query_id,
        conversation_id: conversationId,
        rating,
        answer: msg.content,
        chunk_ids: msg.response.citations.map((c) => c.chunk_id),
      });
    } catch {
      // Revert on failure
      setMessages((prev) =>
        prev.map((m, i) => (i === msgIdx ? { ...m, feedback: null } : m))
      );
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  };

  const isEmpty = messages.length === 0;
  const isLoading = loadingPhase !== null;

  return (
    <div className="flex flex-col h-full bg-slate-950/20 backdrop-blur-sm relative overflow-hidden">
      {/* Scrollable messages container */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-6">
        {isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center space-y-8 animate-fade-in py-12">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-brand-500 to-teal-500 rounded-full blur-2xl opacity-20 group-hover:opacity-40 transition-opacity duration-500" />
              <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-tr from-brand-600 to-teal-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
                <Bot className="w-8 h-8 text-white animate-pulse" />
              </div>
            </div>

            <div className="space-y-3">
              <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
                <span className="text-glow-gradient">Enterprise RAG Copilot</span>
              </h1>
              <p className="text-slate-400 text-sm md:text-base max-w-md mx-auto leading-relaxed">
                Interact with your enterprise documents using secure, audited Retrieval-Augmented Generation.
              </p>
            </div>

            {!hasDocuments && (
              <div className="p-4 rounded-xl border border-amber-500/25 bg-amber-500/5 text-amber-300 text-xs md:text-sm max-w-md mx-auto">
                No documents indexed yet. Upload files in the left sidebar to start asking questions.
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl pt-4">
              {SAMPLE_QUERIES.map((query) => (
                <button
                  key={query}
                  onClick={() => submit(query)}
                  disabled={isLoading || !hasDocuments}
                  className="glass-card hover:glass-card text-left p-3.5 rounded-xl border border-white/5 bg-slate-900/35 text-xs md:text-sm text-slate-300 font-medium hover:text-white transition-all duration-300 disabled:opacity-50 disabled:pointer-events-none"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-4 message-animate ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && (
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-slate-900 border border-white/10 flex items-center justify-center shadow-md">
                    <Bot className="w-4 h-4 text-brand-400" />
                  </div>
                )}

                <div
                  className={`relative group max-w-[85%] rounded-2xl px-4 py-3.5 text-sm leading-relaxed shadow-lg ${
                    msg.role === "user"
                      ? "bg-gradient-to-tr from-brand-600 to-indigo-600 text-white font-medium rounded-tr-none"
                      : "glass-card border border-white/5 text-slate-200 rounded-tl-none bg-slate-900/40"
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="space-y-4">
                      {msg.error ? (
                        <div className="text-rose-400 font-mono text-xs p-2.5 rounded bg-rose-950/20 border border-rose-500/20">
                          {msg.error}
                        </div>
                      ) : (
                        <div className="prose prose-invert max-w-none text-slate-200 prose-sm prose-p:leading-relaxed prose-pre:bg-slate-950/40 prose-pre:border prose-pre:border-white/5">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      )}

                      {/* Metadata: evaluation and citations */}
                      {msg.response && (
                        <div className="border-t border-white/5 pt-3 mt-3 space-y-3">
                          {msg.response.eval && (
                            <EvalBadge
                              eval={msg.response.eval}
                              backend={msg.response.generator_backend}
                              responseTimeMs={msg.response.response_time_ms}
                              timings={msg.response.timings}
                            />
                          )}
                          {msg.response.citations && msg.response.citations.length > 0 && (
                            <CitationCard citations={msg.response.citations} />
                          )}

                          {/* Feedback actions */}
                          {msg.response.query_id && (
                            <div className="flex items-center justify-end gap-2 text-xs text-slate-500 pt-1">
                              <span className="mr-2">Helpful?</span>
                              <button
                                onClick={() => handleFeedback(idx, 1)}
                                disabled={msg.feedback !== null}
                                className={`p-1.5 rounded-lg border transition-all ${
                                  msg.feedback === 1
                                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                                    : "border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5"
                                }`}
                                aria-label="Thumbs up"
                              >
                                <ThumbsUp size={13} />
                              </button>
                              <button
                                onClick={() => handleFeedback(idx, -1)}
                                disabled={msg.feedback !== null}
                                className={`p-1.5 rounded-lg border transition-all ${
                                  msg.feedback === -1
                                    ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                                    : "border-transparent text-slate-500 hover:text-slate-350 hover:bg-white/5"
                                }`}
                                aria-label="Thumbs down"
                              >
                                <ThumbsDown size={13} />
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-550 flex items-center justify-center shadow-md">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {/* Live Streaming Token Output Bubble */}
            {isLoading && (
              <div className="flex gap-4 message-animate justify-start">
                <div className="shrink-0 w-8 h-8 rounded-lg bg-slate-900 border border-white/10 flex items-center justify-center shadow-md">
                  {loadingPhase === "generating" ? (
                    <Bot className="w-4 h-4 text-brand-400 animate-pulse" />
                  ) : (
                    <Loader2 className="w-4 h-4 text-teal-400 animate-spin" />
                  )}
                </div>

                <div className="relative group max-w-[85%] rounded-2xl px-4 py-3.5 text-sm leading-relaxed shadow-lg glass-card border border-white/5 text-slate-200 rounded-tl-none bg-slate-900/40">
                  {loadingPhase !== "generating" ? (
                    <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                      <Loader2 size={12} className="animate-spin text-brand-400" />
                      <span>{loadingPhase}</span>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="prose prose-invert max-w-none text-slate-200 prose-sm prose-p:leading-relaxed">
                        <ReactMarkdown>{streamingContent || "Generating response…"}</ReactMarkdown>
                      </div>
                      <span className="flex h-1.5 w-1.5 relative mt-2 animate-pulse">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-brand-500"></span>
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Floating Input Container */}
      <div className="shrink-0 p-4 md:p-6 bg-gradient-to-t from-slate-950 via-slate-950/90 to-transparent">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
          className="max-w-3xl mx-auto relative glowing-input-container rounded-2xl border border-white/10 bg-slate-900/45 backdrop-blur-md overflow-hidden flex items-end shadow-xl shadow-black/40"
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              !hasDocuments
                ? "Upload documents first to start chatting…"
                : "Ask a question about your documents…"
            }
            disabled={isLoading || !hasDocuments}
            className="flex-1 max-h-32 min-h-[44px] py-3 px-4 bg-transparent border-0 text-slate-100 placeholder-slate-555 text-sm focus:outline-none focus:ring-0 resize-none font-sans leading-relaxed"
          />

          <div className="flex items-center h-[44px] px-3">
            <button
              type="submit"
              disabled={isLoading || !input.trim() || !hasDocuments}
              className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center text-white transition-all duration-300 hover:scale-105 active:scale-95 disabled:opacity-30 disabled:hover:scale-100 disabled:pointer-events-none shadow-md shadow-brand-500/10"
              aria-label="Send message"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
