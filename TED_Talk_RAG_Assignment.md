# Assignment: Build a TED Talk RAG Assistant

## Goal
Create a knowledgeable AI assistant specialized in **TED Talks** using a **Retrieval-Augmented Generation (RAG)** system.  
The assistant must answer questions **strictly and only** using the provided TED dataset (metadata + transcripts), without relying on any external knowledge.

---

## Dataset

You will work with a TED dataset in **English**.

**File:**
- `ted_talks_en.csv`

**Schema (CSV columns):**
- `talk_id`
- `title`
- `speaker_1`
- `all_speakers`
- `occupations`
- `about_speakers`
- `views`
- `recorded_date`
- `published_date`
- `event`
- `native_lang`
- `available_lang`
- `comments`
- `duration`
- `topics`
- `related_talks`
- `url`
- `description`
- `transcript`

---

## Functional Requirements: Query Capabilities

Your RAG system must support the following question types using only retrieved dataset context.

### 1. Precise Fact Retrieval
**Goal:** Locate a single, specific entity or fact based on semantic criteria.

**Example:**
> “Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker.”

---

### 2. Multi-Result Topic Listing (Up to 3 Results)
**Goal:** Return multiple *distinct* talks that match a theme.

**Example:**
> “Which TED talks focus on education or learning? Return exactly 3 talk titles.”

**Constraints:**
- Maximum of 3 results
- Results must be from different talks

---

### 3. Key Idea Summary Extraction
**Goal:** Identify a relevant talk and provide a concise summary of its main idea.

**Example:**
> “Find a TED talk where the speaker talks about technology improving people’s lives. Provide the title and a short summary of the key idea.”

---

### 4. Recommendation with Evidence-Based Justification
**Goal:** Recommend one relevant talk and justify the recommendation using retrieved data.

**Example:**
> “I’m looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?”

---

## Constraints
- The system **must not** rely on common knowledge or external sources.
- If an answer cannot be derived from the provided context, the assistant must respond accordingly.

---

## Tools, Budget & Constraints

### Available Models
- `RPRTHPB-text-embedding-3-small` (1536 dimensions)
- `RPRTHPB-gpt-5-mini`

### Budget
- **Total budget:** 5 USD (development + testing)

### Efficiency Guidelines
- Avoid re-embedding the same data
- Start with a small subset, validate, then scale
- Excessive context usage will be considered suboptimal

---

## RAG Hyperparameters
- **Chunk size:** max 2048 tokens
- **Overlap ratio:** max 0.3
- **Top-k retrieved chunks:** max 30

---

## Required System Prompt

> You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond:  
> **“I don’t know based on the provided TED data.”**  
> Always explain your answer using the given context, quoting or paraphrasing relevant transcript or metadata when helpful.

---

## Vector Database & Deployment

### Vector Database
- **Pinecone**
- Index dimensions must match the embedding model
- Index must remain active until grading is complete

### Deployment
- **Vercel**
- A public live URL must be submitted

---

## API Requirements

### POST `/api/prompt`

**Input:**
```json
{
  "question": "Your natural language question here"
}
```

**Output:**
```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "talk_id": "1234",
      "title": "Sample TED Talk",
      "chunk": "Retrieved transcript chunk",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt used",
    "User": "User prompt used"
  }
}
```

---

### GET `/api/stats`

```json
{
  "chunk_size": 1024,
  "overlap_ratio": 0.2,
  "top_k": 5
}
```

---

## Deliverable & Deadline
- Submit your **public URL**
- **Deadline:** 21.12.2025 (end of day)

---

## Notes
- Start small, validate early
- Avoid re-embedding unnecessarily
- Design for efficiency and cost-awareness
