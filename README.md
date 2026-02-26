# Agri-Advisor Pro

An AI-powered agricultural advisory agent for Israeli farmers, built with LangChain, FastAPI, Pinecone, and Supabase. Deployed on Render.

---

## Project Requirements Coverage

| Requirement | Implementation |
|---|---|
| FastAPI backend | `main.py` — serves all API endpoints |
| LangChain AgentExecutor | `create_openai_tools_agent` with two tools |
| Supabase (PostgreSQL) | Chat and message history persistence |
| Pinecone vector DB | RAG over 13 agricultural PDFs (6,380 chunks) |
| Deployed on Render | Live at `https://agents-960290.onrender.com` |
| `GET /api/team_info` | Group info and student list |
| `GET /api/agent_info` | Agent description, prompt template, worked examples |
| `GET /api/model_architecture` | Architecture diagram (PNG) |
| `POST /api/execute` | Main agent endpoint with steps logging |
| Module names in steps | `AgentLLM`, `WeatherTool`, `AgriKnowledgeBase` |
| Multi-turn conversation | `chat_id` parameter with Supabase history |
| Hebrew responses | System prompt and all agent output in Hebrew |

---

## Efficiency Design — Strengths & Tradeoffs

The implementation addresses the three efficiency requirements: avoiding unnecessary LLM calls, minimising prompt/context size, and staying within budget.

### LLM Call Efficiency

| Decision | Strength | Tradeoff |
|---|---|---|
| **Dedicated `retriever_llm`** for `MultiQueryRetriever` | Isolated from the main agent LLM; `temperature=0`, `streaming=False` — deterministic and fast, no token-streaming overhead. Swappable to a cheaper model independently. | `MultiQueryRetriever` still makes one extra LLM call per RAG search to generate sub-queries; the benefit is significantly better recall over a single-query retriever. |
| **Casual chat skips all tools** | The system prompt instructs the agent not to call any tool for non-agricultural chitchat, saving 1–2 LLM tool calls per casual message. | None — correct by design. |
| **Retry capped at exactly 1** | If `agri_knowledge_base` returns no results, the agent tries once more with different keywords, then proceeds. Prevents unbounded retry loops. | One retry adds a round-trip in edge cases, but guarantees a response and better coverage than zero retries. |
| **Lazy Pinecone indexing** | PDFs are only indexed if the index is empty; all subsequent restarts skip this entirely — no wasted embedding calls. | None. |

### Context / Prompt Size Efficiency

| Decision | Strength | Tradeoff |
|---|---|---|
| **Chat history window: first 4 + last 4 messages** | Hard cap of 8 messages passed to the LLM regardless of conversation length. Preserves topic context (first 2 turns) and recency (last 2 turns) while bounding token cost. | Messages in the middle of a long conversation are dropped. Acceptable for most agricultural Q&A flows where context resets per topic. |
| **Weather output: today + 7-day + 30-day summaries** | Historical climate context is agriculturally necessary — planting, irrigation, and pest decisions depend on recent weather trends, not just today's readings. Always included. | Adds ~30 lines to the agent scratchpad. Justified because the information is directly relevant to every professional query. |
| **RAG chunks labelled as authoritative reference** | Wrapping chunks in `=== AGRICULTURAL KNOWLEDGE BASE — AUTHORITATIVE REFERENCE MATERIAL ===` signals to the LLM what the block is and how to weight it, improving answer quality without extra calls. | Adds a single header line per tool call — negligible cost. |
| **Compact system prompt (8 rules)** | System prompt is intentionally short. Rules cover tool usage, language, retry behaviour, jargon explanation, response-length calibration, location assumptions, and security without verbose prose. | — |
| **`SAFETY_INSTRUCTIONS` variable** | Prompt injection and role-break attempts are rejected in Hebrew, keeping the agent in character. Defined once as a module constant, reused in the prompt. | Adds ~3 lines to the system prompt — a deliberate, minimal cost for robustness. |
| **Weather data cached per city** | Raw hourly JSON (~13–16 MB) is loaded and aggregated to one row per day once. All subsequent lookups for that city return in <2 ms with no file I/O. | Memory usage grows with number of cities queried in a session (bounded to 16 stations). |

---

## Features Beyond Base Requirements

- **Rich weather context** — WeatherTool returns today's conditions plus rolling 7-day and 30-day summaries (avg/max/min temperature, total rainfall, humidity, frost days) for long-term agricultural planning.
- **Smart weather caching** — first query for a city aggregates hourly → daily (~1,800 rows); all subsequent queries for any date in that city return in <2 ms.
- **Background initialisation** — port binds immediately on startup; Supabase, LLM, embeddings, and Pinecone connect in a background thread.
- **Lazy Pinecone indexing** — PDFs are indexed only if the Pinecone index is empty; subsequent restarts skip indexing entirely.
- **Multi-city weather coverage** — 16 Israeli weather stations, matched by a deterministic Hebrew/English alias map with sorted fallback.
- **Architecture viewer in UI** — sidebar button fetches and displays the system architecture diagram inline.
- **Streaming UI** — `/get-advice` endpoint streams tokens in real time; `/api/execute` returns full JSON with traced steps.

---

## Knowledge Base Data

### Agricultural PDFs (`project_sources/`) — 137 MB, 13 files

| File | Subject |
|---|---|
| `Building-Soils-for-Better-Crops.pdf` | Soil health, organic matter, fertility management |
| `Managing-Cover-Crops-Profitably.pdf` | Cover crop selection, planting, integration |
| `cc3338en.pdf` | FAO good agricultural practices (GAP) guide |
| `einboeck_source_1.pdf` | Mechanical weed control in field crops |
| `source2.pdf` – `source12.pdf` | Additional agronomy sources (irrigation, diseases, soil, water management in Israel) |

All PDFs are chunked (1,000 tokens, 200 overlap) and embedded with `BAAI/bge-small-en-v1.5` (384 dimensions) into Pinecone. **6,380 vectors** total.

### Weather Data (`city_data/`) — 212 MB, 16 files

Hourly meteorological readings from 16 Israeli weather stations covering 2025. Columns: `date`, `TD`, `TDmax`, `TDmin`, `RH`, `Rain`, `WS`, `WSmax`, `WD`, `STDwd`, and others.

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
        ├─ Retrieve chat history from Supabase (if chat_id)
        │   └─ Window: first 4 + last 4 messages (≤8 total)
        │
        ▼
LangChain AgentExecutor
        │
        ├─── AgentLLM decides which tools to call
        │
        ├─── WeatherTool (weather/climate questions)
        │         └─ City daily cache (<2ms if warm)
        │             Return: today + 7-day + 30-day summaries
        │
        ├─── AgriKnowledgeBase (agronomic questions)
        │         └─ retriever_llm (temp=0) generates 3 sub-queries
        │             → Pinecone vector search (k=3 each)
        │             → Return chunks labelled as authoritative reference
        │             → Retry once with different keywords if no results
        │
        └─── AgentLLM synthesises final Hebrew response
                │
                ▼
        Save user + bot messages to Supabase (if chat_id)
                │
                ▼
        Return JSON:
        {
          "status": "ok",
          "error": null,
          "response": "...",   ← Hebrew answer
          "steps": [           ← one entry per tool call / LLM step
            {"module": "WeatherTool",      "prompt": "...", "response": "..."},
            {"module": "AgriKnowledgeBase","prompt": "...", "response": "..."},
            {"module": "AgentLLM",         "prompt": "...", "response": "..."}
          ]
        }
```

---

## Setup on Render

### Prerequisites
- [Render](https://render.com) account
- [Supabase](https://supabase.com) project with the tables below
- [Pinecone](https://pinecone.io) project with an `agri-advisor` index (dimension 384, cosine)
- An OpenAI-compatible LLM API key (project uses [llmod.ai](https://llmod.ai))

### 1. Supabase tables

```sql
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_name TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    chat_id TEXT,
    role TEXT,
    content TEXT
);
ALTER TABLE chats DISABLE ROW LEVEL SECURITY;
ALTER TABLE messages DISABLE ROW LEVEL SECURITY;
```

### 2. Render web service

1. Create a new **Web Service** and connect this repository.
2. Set environment variables in the Render dashboard:

| Variable | Value |
|---|---|
| `LLMOD_API_KEY` | Your LLM API key |
| `LLMOD_API_BASE` | `https://api.llmod.ai/v1` (or your endpoint) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anon/service key |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `agri-advisor` |

3. Render will use `render.yaml` automatically.
4. On first deploy, PDFs are indexed into Pinecone (one-time, ~2–5 minutes). Subsequent deploys skip this.

### 3. Populate Pinecone locally (recommended)

```bash
pip install -r requirements.txt
# set env vars in .env, then run build_index() from main.py
```

Or deploy and wait — the agent responds normally while indexing runs in the background.
