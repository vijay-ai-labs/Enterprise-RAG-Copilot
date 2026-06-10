# Retrieval-Augmented Generation (RAG): A Complete Guide

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Models (LLMs) by giving them access to external knowledge at inference time. Rather than relying solely on knowledge encoded in model weights during training, RAG systems retrieve relevant documents from a knowledge base and provide them as context to the LLM when generating responses.

RAG was introduced by Lewis et al. in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" from Facebook AI Research. It combines the parametric knowledge stored in LLM weights with the non-parametric knowledge stored in a retrieval corpus.

## Why RAG?

LLMs have several fundamental limitations that RAG addresses:

**Knowledge Cutoff**: LLMs are trained on data up to a cutoff date and cannot access information about events after training. RAG allows the model to answer questions about current events by retrieving up-to-date documents.

**Hallucination**: LLMs sometimes generate plausible-sounding but factually incorrect information. By grounding responses in retrieved documents, RAG significantly reduces hallucination rates.

**Source Attribution**: RAG systems can cite the specific documents used to generate an answer, enabling users to verify claims and trace information to its source.

**Domain Specialization**: RAG allows a general-purpose LLM to become an expert on a specific corpus (company documents, legal texts, medical literature) without fine-tuning.

**Cost Efficiency**: Adding knowledge via retrieval is much cheaper than retraining or fine-tuning a model on new data.

## RAG Architecture

A production RAG system consists of two main pipelines:

### 1. Ingestion Pipeline (Offline)

The ingestion pipeline processes documents and builds the retrieval index:

```
Raw Documents (PDF, TXT, MD, HTML)
        ↓
   Text Extraction
        ↓
   Text Chunking (with overlap)
        ↓
   Embedding Generation
        ↓
   Vector Store (FAISS, Pinecone, Weaviate, Chroma)
```

**Text Extraction**: Documents are parsed and raw text is extracted. For PDFs, page-level extraction preserves structure and enables accurate citation.

**Chunking**: Long documents are split into smaller chunks (typically 256–1024 tokens) with overlap (10–20% of chunk size). Overlap prevents context from being lost at chunk boundaries. Each chunk stores metadata: source filename, page number, chunk index, and timestamp.

**Embedding**: Each chunk is encoded into a dense vector using an embedding model. Sentence-transformers (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2) are popular open-source choices. OpenAI's text-embedding-3-small offers higher quality at a small cost.

**Vector Store**: Embeddings are stored in a vector database optimized for approximate nearest neighbor (ANN) search. FAISS (Facebook AI Similarity Search) is a popular in-memory option. For production, Pinecone, Weaviate, Qdrant, or pgvector provide managed solutions.

### 2. Inference Pipeline (Online)

The inference pipeline handles user queries in real time:

```
User Query
     ↓
Query Embedding
     ↓
Vector Search (Top-K retrieval)
     ↓
Similarity Threshold Filtering
     ↓
Context Assembly
     ↓
LLM Prompt Construction
     ↓
Answer Generation
     ↓
Citation Extraction
     ↓
Response to User
```

**Query Embedding**: The user's question is encoded using the same embedding model used during ingestion. Consistency is critical — different models produce incompatible vector spaces.

**Vector Search**: The query vector is compared against all stored chunk vectors using cosine similarity (or inner product on normalized vectors). Top-K most similar chunks are retrieved. FAISS supports both exact search (IndexFlatIP) and approximate search (IndexIVFFlat, HNSW) for scalability.

**Threshold Filtering**: Retrieved chunks below a similarity threshold (e.g., 0.3) are discarded. This prevents low-quality context from degrading the answer.

**Context Assembly**: Retrieved chunks are formatted and inserted into the LLM prompt. The prompt template instructs the model to answer based only on the provided context.

**Answer Generation**: The LLM generates a response grounded in the retrieved context. A well-designed prompt includes: the context chunks (with source labels), the user question, and an instruction to cite sources and say "I don't know" if the context is insufficient.

## Chunking Strategies

Chunking strategy significantly impacts retrieval quality:

**Fixed-size chunking**: Split text every N characters with M-character overlap. Simple and fast, but may break sentences mid-way.

**Sentence-level chunking**: Split at sentence boundaries using NLP tools (spaCy, NLTK). Produces more coherent chunks.

**Semantic chunking**: Split where embedding similarity between adjacent sentences drops below a threshold. Produces topically coherent chunks but is computationally expensive.

**Hierarchical chunking**: Store both paragraph-level and document-level summaries. Retrieve at multiple granularities and merge results.

## Embedding Models

The choice of embedding model determines retrieval quality:

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Very fast | Good |
| all-mpnet-base-v2 | 768 | Fast | Better |
| text-embedding-3-small | 1536 | API call | Very good |
| text-embedding-3-large | 3072 | API call | Best |

For production with budget constraints, all-MiniLM-L6-v2 offers an excellent speed-quality tradeoff. It runs on CPU with reasonable latency.

## RAG Evaluation

Evaluating RAG systems requires assessing multiple dimensions:

**Retrieval Quality**:
- Context Relevance: Are the retrieved chunks actually related to the query?
- Recall: Are all relevant documents retrieved?

**Generation Quality**:
- Faithfulness/Groundedness: Is the answer supported by the retrieved context?
- Answer Relevance: Does the answer address the user's question?
- Citation Accuracy: Are the cited sources correct?

Popular evaluation frameworks:
- **RAGAS**: Open-source RAG evaluation framework using LLMs to score faithfulness, answer relevance, context relevance, and context recall.
- **TruLens**: Evaluation and tracking for LLM applications with RAG-specific metrics.
- **Arize Phoenix**: LLM observability with built-in RAG evaluation.

## Advanced RAG Techniques

**HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer to the query, embed it, and use that embedding for retrieval. Often retrieves more relevant documents than embedding the query directly.

**Query Rewriting**: Use an LLM to rewrite the user query to be more retrieval-friendly before embedding.

**Re-ranking**: After initial retrieval, apply a cross-encoder re-ranker (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to more accurately score chunk relevance. More expensive but significantly improves precision.

**Multi-query Retrieval**: Generate multiple query variations and merge the retrieved sets. Improves recall for ambiguous queries.

**Contextual Compression**: Extract only the relevant portion of each retrieved chunk rather than using the full chunk. Reduces context length and noise.

**Parent Document Retrieval**: Index small chunks for precise retrieval, but return the parent document section for richer context.

## Common Pitfalls

1. **Chunk size too large**: Retrieves noisy context that confuses the LLM
2. **Chunk size too small**: Lacks sufficient context for meaningful retrieval
3. **No overlap between chunks**: Loses context at boundaries
4. **Wrong similarity threshold**: Too high misses relevant results; too low adds noise
5. **Mismatched embedding models**: Using different models for ingestion and query
6. **No re-ranking**: First-stage retrieval precision is often insufficient
7. **Context window overflow**: Too many chunks exceed the LLM's context limit
8. **No evaluation**: Shipping RAG without measuring retrieval and generation quality

## Production Considerations

**Scalability**: FAISS works well for millions of documents in memory. For larger scales, use distributed vector databases (Pinecone, Qdrant, Weaviate) with sharding.

**Freshness**: Implement incremental indexing to add new documents without rebuilding the entire index.

**Security**: Implement access control at the retrieval layer — only return chunks the user is authorized to see.

**Latency**: Profile each stage. Embedding the query is fast (10–50ms). FAISS search is fast (1–10ms). LLM generation dominates (500ms–10s depending on model and length).

**Caching**: Cache embeddings for repeated queries. Cache LLM responses for identical query+context combinations.

**Monitoring**: Log every query, retrieved chunks, response, and evaluation scores. Track retrieval hit rate, average similarity scores, and user feedback over time.
