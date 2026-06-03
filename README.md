# Agri-Advisor Pro

An AI-powered agricultural advisory agent for Israeli farmers, built with **LangChain** and **FastAPI**. Runs **fully locally** on [Ollama](https://ollama.com) — no external API keys required.

> **Local-first rebuild.** The project originally used a hosted LLM (llmod.ai gpt-5-mini), Pinecone, and Supabase. It now runs entirely on local infrastructure: the LLM and embeddings are served by Ollama, the vector store is a local FAISS index, and chat history lives in a local SQLite file. This removes all API-key dependencies and the network round-trips that dominated response time.

---

## Local Stack

| Component | Technology |
|---|---|
| LLM (agent + Hebrew→English query expansion) | Ollama — `llama3.1:8b` |
| Embeddings | Ollama — `nomic-embed-text` (768-dim) |
| Vector store | Local **FAISS** index (`faiss_nomic_index/`) |
| Chat history | Local **SQLite** (`agri_advisor.db`) |
| Web framework | FastAPI |
| Agent | LangChain `create_openai_tools_agent` + `AgentExecutor` with two tools |

All model names are overridable via environment variables: `OLLAMA_AGENT_MODEL`, `OLLAMA_RETRIEVER_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_BASE_URL`.

---

## Project Requirements Coverage

| Requirement | Implementation |
|---|---|
| FastAPI backend | `main.py` — serves all API endpoints |
| LangChain AgentExecutor | `create_openai_tools_agent` with two tools |
| Chat / message persistence | Local SQLite (`db_service.py`) |
| Vector DB (RAG) | Local FAISS over agricultural PDFs |
| `GET /api/team_info` | Group info and student list |
| `GET /api/agent_info` | Agent description, prompt template, worked examples |
| `GET /api/model_architecture` | Architecture diagram (PNG) |
| `POST /api/execute` | Main agent endpoint with steps logging |
| Module names in steps | `AgentLLM`, `WeatherTool`, `AgriKnowledgeBase` |
| Multi-turn conversation | `chat_id` parameter with SQLite history |
| Hebrew responses | System prompt and all agent output in Hebrew |

---

## What Was Improved (vs. the reviewed version)

The previous version had several issues that this rebuild addresses directly:

### 1. Be'er Sheva weather returned N/A for everything
**Root cause:** the Be'er Sheva station only records **ground temperature (`TG`)** and barometric pressure — it never reports air temperature (`TD`), humidity, wind, or rain (verified: 0 of 18,185 readings). The old code only looked at `TD`, so every field showed N/A.
**Fix (`weather_service.py`):** when air temperature is unavailable, the tool falls back to **ground temperature, clearly labelled as such**, and shows honest `N/A` only for sensors the station genuinely lacks (instead of a misleading `0.0 mm`).

### 2. Knowledge-base retrieval returned metadata, not practical content
**Root cause:** two problems compounded. (a) Hebrew queries were embedded with an **English-only** model (`bge-small-en-v1.5`), so a Hebrew question for "wheat sowing" retrieved unrelated chunks that merely contained the word "Israel" (water-law, desalination, finance). (b) Boilerplate chunks (book endorsements, tables of contents, author bios, reference lists) competed with real agronomic content.
**Fix (`rag_service.py`):**
- **Hebrew→English query expansion** — every question is expanded by the LLM into 3 English search queries before embedding, matching the language of the (English) source manuals. The original query is also searched as a safety net.
- **Junk-chunk filter at index time** — front-matter / boilerplate is dropped (≈22% of chunks), so retrieval surfaces practical content.
- **`nomic-embed-text` embeddings** via Ollama, replacing the hosted English-only model.

### RAG audit — additional retrieval fixes

A deeper audit of common RAG failure modes surfaced three more issues, all fixed:

- **Missing embedding task prefixes.** `nomic-embed-text` is trained to require `search_document:` on indexed text and `search_query:` on queries; LangChain's `OllamaEmbeddings` does not add them. Adding them (`NomicPrefixedEmbeddings`) **dramatically tightened retrieval** — relevant-chunk L2 distances dropped from ~1.5 to ~0.33, with a clean gap between relevant (~0.3–0.5) and off-topic (~0.73+) results.
- **Duplicate source content.** `source2.pdf` is a near-duplicate of the FAO guide `cc3338en.pdf` (same passages, e.g. the gypsum text). Identical chunks are now de-duplicated at index time so they don't occupy multiple retrieval slots.
- **No relevance threshold.** Retrieval now drops chunks with L2 distance > 0.70, so sparse/irrelevant queries return *no results* instead of tangential text.

### Grounding — is it faithful?

Verified directly: answers are **grounded in the retrieved chunks, not hallucinated**. For example, a specific figure like "gypsum (CaSO₄·2H₂O) 5 t/ha for correcting soil sodicity" traces verbatim to `cc3338en.pdf`. Caveats inherent to a local 8B model: (a) Hebrew phrasing is occasionally awkward or mistranslates a term (the underlying fact is still from the source); (b) retrieval precision isn't perfect — a broad query can pull a related-but-not-exact chunk (e.g. a general cover-crop list for a "nitrogen-fixing" question). Some sources (`cc3338en`, `source2`) are India-centric FAO guides, so a few recommendations (crop names, metric dosages) reflect that context rather than Israel specifically.

### 3. Response times
**Fix:** all LLM, embedding, and vector-search calls are now **local** — no llmod.ai / Pinecone network latency. The agent runs at `temperature=0` (deterministic tool use), answer length is capped (`num_predict`), the context window is bounded (`num_ctx`), tool-loop iterations are limited (`max_iterations=4`), RAG returns only the top 3 chunks, and the fast 3B model handles the short query-expansion step.

Typical latency on a local M-series Mac: **weather questions ≈ 10–25 s; knowledge-base questions ≈ 60–90 s**. The remaining cost is inherent to generating a Hebrew answer with a local 8-billion-parameter model (the smaller, faster 3B model was tested but degenerates on Hebrew, so it is used only for the English expansion step). On a machine with a GPU, or by pointing `OLLAMA_*` env vars at a larger/faster served model, latency drops further with no code changes.

### 4. Reliability safety net
Small local models occasionally refuse to answer even after a tool returned valid data. `_repair_if_spurious_refusal` in `main.py` detects this and re-synthesises a grounded Hebrew answer directly from the collected tool data (the path the local model handles reliably). For weather/agriculture questions it will, if needed, gather the tool data deterministically — guaranteeing a grounded answer instead of a spurious "no information found".

---

## Knowledge Base Data

### Agricultural PDFs (`project_sources/`) — 12 files

| File | Subject |
|---|---|
| `Building-Soils-for-Better-Crops.pdf` | Soil health, organic matter, fertility management |
| `Managing-Cover-Crops-Profitably.pdf` | Cover crop selection, planting, integration |
| `cc3338en.pdf` | FAO good agricultural practices (GAP) guide |
| `einboeck_source_1.pdf` | Mechanical weed control in field crops |
| `source2.pdf` – `source12.pdf` | Additional agronomy sources (irrigation, diseases, soil, water management in Israel) |

PDFs are chunked (900 tokens, 150 overlap), filtered for boilerplate, and embedded with `nomic-embed-text` (768 dimensions) into a local FAISS index — **4,486 vectors** after filtering.

### Weather Data (`city_data/`) — 16 files

Hourly meteorological readings from 16 Israeli weather stations covering 2025. Columns include `date`, `TD` (air temp), `TG` (ground temp), `RH`, `Rain`, `WS`, `WD`, and others. **Not every station reports every sensor** — Be'er Sheva, for example, only reports ground temperature and pressure.

| Station | Station |
|---|---|
| Ariel | Ashdod |
| Ashkelon | Avne Eitan |
| Beer Sheva | Eilat |
| Hadera | Haifa Technion |
| Haifa Bate Zakuk | Jerusalem Center |
| Lev Kineret | Maale Gilboa |
| Nitzan | Tel Aviv Beach |
| Yotvata | Zichron Yaakov |

---

## System Pipeline

```
User message (prompt + city + date + optional chat_id)
        │
        ▼
POST /api/execute  (FastAPI)
        │
        ├─ Retrieve chat history from SQLite (if chat_id)
        │   └─ Window: first 4 + last 4 messages (≤8 total)
        │
        ▼
LangChain AgentExecutor  (Ollama llama3.1:8b, temp=0, max_iterations=4)
        │
        ├─── AgentLLM decides which tools to call
        │
        ├─── WeatherTool (weather/climate questions)
        │         └─ City daily cache (<2 ms if warm)
        │             Air temp, or ground-temp fallback if no air sensor
        │             Return: today + 7-day + 30-day summaries
        │
        ├─── AgriKnowledgeBase (agronomic questions)
        │         └─ LLM expands question → 3 English sub-queries
        │             → FAISS vector search (k=4 each) + verbatim query
        │             → dedupe, drop junk, return top chunks as reference
        │
        └─── AgentLLM synthesises final Hebrew response
                │
                ├─ Safety net: if a spurious refusal is detected,
                │   re-synthesise directly from the collected tool data
                │
                ▼
        Save user + bot messages to SQLite (if chat_id)
                │
                ▼
        Return JSON { status, error, response, steps[] }
```

---

## Setup (Local)

### Prerequisites
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Python 3.11+

### 1. Pull the models
```bash
ollama pull llama3.1:8b          # main agent + Hebrew answers
ollama pull llama3.2             # fast English query-expansion
ollama pull nomic-embed-text     # embeddings
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python main.py            # serves on http://localhost:8000
```

On first run, `rag_service.build_or_load` extracts the PDFs, filters boilerplate, embeds the chunks with `nomic-embed-text`, and saves the FAISS index to `faiss_nomic_index/` (one-time, ~2–3 minutes). Subsequent runs load the saved index instantly. The SQLite chat store (`agri_advisor.db`) is created automatically.

### Optional environment variables
| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_AGENT_MODEL` | `llama3.1:8b` | Main tool-calling agent |
| `OLLAMA_RETRIEVER_MODEL` | `llama3.1:8b` | Hebrew→English query expansion |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embeddings |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `PORT` | `8000` | HTTP port |

---

## Files

| File | Role |
|---|---|
| `main.py` | FastAPI app, agent setup, endpoints, refusal safety net |
| `rag_service.py` | FAISS index build/load, junk filter, Hebrew→English retriever |
| `weather_service.py` | City weather aggregation with ground-temp fallback |
| `db_service.py` | Local SQLite chat store |
| `generate_architecture.py` | Regenerates `static/architecture.png` |
| `test_project.py` | Test suite |
