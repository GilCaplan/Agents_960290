"""
Local RAG service for the agricultural advisor.

Replaces the previous Pinecone + bge-small-en-v1.5 stack (which required an API
key and embedded the *English* knowledge base with a model that could not encode
the Hebrew user queries) with a fully local pipeline:

  * Embeddings : Ollama `nomic-embed-text` (768-dim) — runs locally, no API key.
  * Vector DB  : local FAISS index on disk.
  * Quality    : front-matter / boilerplate chunks (book endorsements, tables of
                 contents, reference lists, author bios) are filtered out at index
                 time, so retrieval returns practical agronomic content instead of
                 metadata.
  * Hebrew gap : the knowledge base is 100% English, so a Hebrew question is first
                 expanded into several English search queries by a small local LLM
                 before embedding. This is the core fix for the "Hebrew query
                 returns random Israel-mentioning chunks" failure.
"""

import os
import re
import glob
import hashlib
import fitz  # PyMuPDF

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = "faiss_nomic_index"


# --- nomic-embed-text task prefixes ---------------------------------------
# nomic-embed-text is trained with task-instruction prefixes: documents must be
# embedded as "search_document: ..." and queries as "search_query: ...". Omitting
# them measurably degrades retrieval. LangChain's OllamaEmbeddings does NOT add
# them, so we wrap it.
class NomicPrefixedEmbeddings(Embeddings):
    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts):
        return self.base.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return self.base.embed_query(f"search_query: {text}")

# --- Chunk-quality filter -------------------------------------------------
# The feedback was that retrieval mostly surfaced metadata (author names,
# acknowledgments) rather than practical agricultural content. These heuristics
# drop boilerplate at index time so only substantive agronomic chunks are stored.

_JUNK_PATTERNS = re.compile(
    r'all rights reserved|copyright ©|copyright \d|\bISBN\b|table of contents'
    r'|acknowledgment|acknowledgement|cooperative extension'
    r'|president and ceo|valuable source of knowledge|garden of eden'
    r'|^\s*references\s*$|bibliography',
    re.I,
)

_AGRI_TERMS = re.compile(
    r'soil|crop|plant|seed|sow|irrig|water|fertil|nitrogen|compost|manure|pest|'
    r'disease|weed|harvest|yield|root|nutrient|tillage|cover crop|rotation|'
    r'temperature|drainage|organic matter|mulch|germinat|cultivar|grow|field',
    re.I,
)


def is_junk(text: str) -> bool:
    """True if a chunk is boilerplate/front-matter rather than agronomic content."""
    t = text.strip()
    if len(t) < 250:                       # captions, headers, single TOC lines
        return True
    lines = [l for l in t.split('\n') if l.strip()]
    if lines:
        short = sum(1 for l in lines if len(l.strip()) < 35)
        if short / len(lines) > 0.6:       # mostly short lines → TOC / index / name lists
            return True
    agri_hits = len(_AGRI_TERMS.findall(t))
    if agri_hits < 2:                      # no real agricultural substance
        return True
    if _JUNK_PATTERNS.search(t) and agri_hits < 5:   # front-matter with little content
        return True
    return False


# --- Index build / load ---------------------------------------------------
def build_or_load(embeddings, pdf_glob="project_sources/*.pdf"):
    """Load the FAISS index from disk, or build it from the PDFs if absent."""
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print(f"[RAG] Loading existing FAISS index from '{INDEX_DIR}'...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("[RAG] No index found — extracting PDFs...")
    docs = []
    for pdf in glob.glob(pdf_glob):
        src = os.path.basename(pdf)
        doc = fitz.open(pdf)
        for page in doc:
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": src}))
        doc.close()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(docs)

    # Drop junk (boilerplate) and near-duplicate chunks. Several sources overlap
    # heavily (e.g. source2.pdf duplicates the cc3338en FAO guide), so the same
    # passage would otherwise occupy multiple retrieval slots.
    kept, seen = [], set()
    n_junk = n_dup = 0
    for c in chunks:
        if is_junk(c.page_content):
            n_junk += 1
            continue
        norm = re.sub(r'\W+', ' ', c.page_content.lower()).strip()
        sig = hashlib.md5(norm[:300].encode()).hexdigest()   # near-dup key on first 300 chars
        if sig in seen:
            n_dup += 1
            continue
        seen.add(sig)
        kept.append(c)
    print(f"[RAG] {len(chunks)} chunks → {len(kept)} kept "
          f"({n_junk} junk, {n_dup} duplicates dropped). Embedding with nomic-embed-text...")

    vs = FAISS.from_documents(kept, embeddings)
    vs.save_local(INDEX_DIR)
    print(f"[RAG] Index built and saved to '{INDEX_DIR}'.")
    return vs


# --- Retriever with Hebrew → English query expansion ----------------------
_EXPANSION_PROMPT = (
    "You are a search assistant for an ENGLISH agricultural knowledge base "
    "(soil, crops, irrigation, pests, cover crops, weed control). "
    "The user question may be in Hebrew. Translate the agronomic intent and output "
    "exactly 3 short ENGLISH search queries, one per line, no numbering, no extra text.\n\n"
    "User question: {q}\n\nEnglish search queries:"
)


class AgriRetriever:
    """
    Expands a (possibly Hebrew) question into English search queries, runs each
    against the FAISS index, de-duplicates, and returns formatted reference text.
    """

    def __init__(self, vectorstore, llm, k_per_query=3, max_chunks=3,
                 max_distance=0.70):
        self.vs = vectorstore
        self.llm = llm
        self.k_per_query = k_per_query
        self.max_chunks = max_chunks
        # L2 distance over nomic (prefixed) embeddings: relevant agricultural chunks
        # score ~0.3-0.5, off-topic ~0.73+. Drop anything above this so irrelevant
        # queries return NO RESULTS instead of tangential text.
        self.max_distance = max_distance

    @staticmethod
    def _looks_like_meta(line: str) -> bool:
        """Drop preamble the model sometimes emits instead of a query."""
        low = line.lower()
        meta = ("user's question", "search quer", "i'll translate", "i will translate",
                "here are", "english search", "agricultural knowledge base")
        return line.endswith(":") or any(m in low for m in meta)

    def _english_queries(self, question: str):
        try:
            resp = self.llm.invoke(_EXPANSION_PROMPT.format(q=question))
            text = resp.content if hasattr(resp, "content") else str(resp)
            queries = [re.sub(r'^[\d\.\-\)\s]+', '', l).strip()
                       for l in text.splitlines() if l.strip()]
            queries = [q for q in queries if len(q) > 3 and not self._looks_like_meta(q)][:3]
        except Exception as e:
            print(f"[RAG] query expansion failed ({e}); using raw query")
            queries = []
        if not queries:
            queries = [question]
        # Always also search the original question verbatim as a safety net
        if question not in queries:
            queries.append(question)
        return queries

    def search(self, question: str) -> str:
        queries = self._english_queries(question)
        print(f"[RAG] expanded queries: {queries}")

        seen, picked = set(), []
        for q in queries:
            for doc, score in self.vs.similarity_search_with_score(q, k=self.k_per_query):
                key = doc.page_content[:120]
                if key in seen or is_junk(doc.page_content) or score > self.max_distance:
                    continue
                seen.add(key)
                picked.append((score, doc))

        if not picked:
            return ("NO RESULTS FOUND in the knowledge base. "
                    "Try once more with different, broader agricultural keywords.")

        picked.sort(key=lambda x: x[0])          # lower L2 distance = more relevant
        top = picked[: self.max_chunks]

        header = (
            "=== AGRICULTURAL KNOWLEDGE BASE — AUTHORITATIVE REFERENCE MATERIAL ===\n"
            "Answer ONLY from the excerpts below. Do NOT add any number, dosage, "
            "chemical name, cultivar, or fact that is not written here. If a specific "
            "detail is not in the text, leave it out.\n\n"
        )
        body = "\n\n---\n\n".join(
            f"[source: {d.metadata.get('source', '?')}]\n{d.page_content.strip()}"
            for _, d in top
        )
        return header + body
