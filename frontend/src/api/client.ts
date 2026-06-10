import axios from "axios";

const apiKey = import.meta.env.VITE_API_KEY;

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "/",
});

if (apiKey) {
  http.defaults.headers.common["X-API-Key"] = apiKey;
}

// ------------------------------------------------------------------
// Types (mirror app/schemas.py)
// ------------------------------------------------------------------

export interface Citation {
  filename: string;
  page: number;
  chunk_id: string;
  score: number;
  text_snippet: string;
}

export interface EvalResult {
  context_relevance: number;
  answer_groundedness: number;
  citation_presence: number;
  overall: number;
}

export interface IngestTimings {
  extraction_ms: number;
  chunking_ms: number;
  embedding_ms: number;
  indexing_ms: number;
  total_ms: number;
}

export interface QueryTimings {
  embedding_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  evaluation_ms: number;
  total_ms: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  eval: EvalResult;
  generator_backend: string;
  response_time_ms: number;
  query_id?: string;
  conversation_id?: string;
  trace_id?: string;
  timings?: QueryTimings;
  confidence?: number;
  input_tokens?: number;
  output_tokens?: number;
}

export interface IngestResponse {
  filename: string;
  doc_id?: string;
  chunks_stored: number;
  message: string;
  pages_extracted?: number;
  timings?: IngestTimings;
}

export interface HealthResponse {
  status: string;
  version: string;
  index_size: number;
  embed_model: string;
  hybrid_search: boolean;
  reranker: boolean;
}

export interface CacheStats {
  enabled: boolean;
  size: number;
  hits: number;
  misses: number;
  hit_rate: number;
  threshold: number;
  max_size: number;
}

export interface LatencyPercentiles {
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  count: number;
}

export interface MetricsResponse {
  index_size: number;
  total_queries: number;
  avg_response_time_ms: number;
  avg_groundedness: number;
  generator_backend_status: {
    ollama: string;
    flan_t5: string;
    openai: string;
    active_mode: string;
  };
  cache?: CacheStats;
  latency_percentiles?: LatencyPercentiles;
}

export interface MetricsTimeseriesPoint {
  timestamp: number;
  queries: number;
  avg_latency_ms: number;
  avg_groundedness: number;
  total_tokens: number;
}

export interface MetricsTimeseriesResponse {
  points: MetricsTimeseriesPoint[];
  hours: number;
  bucket_minutes: number;
}

export interface ConversationSummary {
  conversation_id: string;
  turn_count: number;
  last_active: number;
  preview: string;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  total: number;
}

export interface ConversationTurn {
  role: string;
  content: string;
  timestamp: number;
}

export interface ConversationHistoryResponse {
  conversation_id: string;
  turns: ConversationTurn[];
}

export interface IngestTaskStatus {
  task_id: string;
  status: "pending" | "processing" | "done" | "error";
  filename?: string;
  doc_id?: string;
  chunks_stored?: number;
  error?: string;
  created_at: number;
  completed_at?: number;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  pages_extracted: number;
  chunks_stored: number;
  file_size_bytes: number;
  ingested_at: number;
  status: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface FeedbackRequest {
  query_id: string;
  conversation_id?: string;
  rating: 1 | -1;
  comment?: string;
  answer?: string;
  chunk_ids?: string[];
}

export interface FeedbackResponse {
  feedback_id: string;
  message: string;
}

// ------------------------------------------------------------------
// SSE stream types
// ------------------------------------------------------------------

export type StreamEvent =
  | { event: "retrieving" }
  | { event: "generating" }
  | { event: "token"; token: string }
  | { event: "evaluating" }
  | { event: "done"; data: QueryResponse }
  | { event: "error"; message: string };

// ------------------------------------------------------------------
// API calls
// ------------------------------------------------------------------

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>("/health");
  return data;
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const { data } = await http.get<MetricsResponse>("/metrics");
  return data;
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await http.post<IngestResponse>("/ingest", form);
  return data;
}

export async function fetchDocuments(): Promise<DocumentListResponse> {
  const { data } = await http.get<DocumentListResponse>("/documents");
  return data;
}

export async function deleteDocument(docId: string): Promise<void> {
  await http.delete(`/documents/${docId}`);
}

export async function submitFeedback(req: FeedbackRequest): Promise<FeedbackResponse> {
  const { data } = await http.post<FeedbackResponse>("/feedback", req);
  return data;
}

export async function queryDocuments(
  query: string,
  topK?: number,
  conversationId?: string,
  sourceFilter?: string[],
): Promise<QueryResponse> {
  const { data } = await http.post<QueryResponse>("/query", {
    query,
    top_k: topK,
    conversation_id: conversationId,
    source_filter: sourceFilter,
  });
  return data;
}

export async function fetchMetricsTimeseries(
  hours = 24,
  bucketMinutes = 60,
): Promise<MetricsTimeseriesResponse> {
  const { data } = await http.get<MetricsTimeseriesResponse>(
    `/metrics/timeseries?hours=${hours}&bucket_minutes=${bucketMinutes}`,
  );
  return data;
}

export async function fetchConversations(limit = 50): Promise<ConversationListResponse> {
  const { data } = await http.get<ConversationListResponse>(`/conversations?limit=${limit}`);
  return data;
}

export async function fetchConversation(id: string): Promise<ConversationHistoryResponse> {
  const { data } = await http.get<ConversationHistoryResponse>(`/conversations/${id}`);
  return data;
}

export async function deleteConversation(id: string): Promise<void> {
  await http.delete(`/conversations/${id}`);
}

export async function ingestFileAsync(file: File): Promise<IngestTaskStatus> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await http.post<IngestTaskStatus>("/ingest/async", form);
  return data;
}

export async function pollIngestTask(taskId: string): Promise<IngestTaskStatus> {
  const { data } = await http.get<IngestTaskStatus>(`/ingest/tasks/${taskId}`);
  return data;
}

export async function clearCache(): Promise<{ message: string }> {
  const { data } = await http.post<{ message: string }>("/cache/clear");
  return data;
}

/**
 * Stream a query via SSE (POST /query/stream).
 * Returns AbortController so the caller can cancel mid-stream.
 */
export function streamQuery(
  query: string,
  topK: number | undefined,
  onEvent: (event: StreamEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
  conversationId?: string,
  sourceFilter?: string[],
): AbortController {
  const controller = new AbortController();
  const baseURL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

  fetch(`${baseURL}/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
    },
    body: JSON.stringify({
      query,
      top_k: topK,
      conversation_id: conversationId,
      source_filter: sourceFilter,
    }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        onError(new Error(`HTTP ${res.status}`));
        return;
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const parsed = JSON.parse(line.slice(6)) as StreamEvent;
              onEvent(parsed);
              if (parsed.event === "done" || parsed.event === "error") {
                onDone();
                return;
              }
            } catch {
              // Ignore malformed lines
            }
          }
        }
      }
      onDone();
    })
    .catch((err: Error) => {
      if (err.name !== "AbortError") onError(err);
    });

  return controller;
}
