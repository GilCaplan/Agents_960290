from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv

# Load env vars (for local testing)
load_dotenv()

# --- CONFIGURATION ---
CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2
TOP_K = 5

# Custom Model Config
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
INDEX_NAME = "ted-rag-index"

app = FastAPI()

#  CLIENT INITIALIZATION
# Embeddings (for query)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.environ.get("LLMOD_API_KEY"),
    base_url=LLMOD_BASE_URL
)

# vector Store (Read-only connection)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# 3. Chat Model
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=1,
    api_key=os.environ.get("LLMOD_API_KEY"),
    base_url=LLMOD_BASE_URL
)

#  SYSTEM PROMPT
SYSTEM_PROMPT_TEXT = """You are a TED Talk assistant that answers questions strictly and 
only based on the TED dataset context provided to you (metadata 
and transcript passages).
You must not use any external 
knowledge, the open internet, or information that is not explicitly 
contained in the retrieved context.
If the answer cannot be 
determined from the provided context, respond: "I don't know 
based on the provided TED data."
Always explain your answer 
using the given context, quoting or paraphrasing the relevant 
transcript or metadata when helpful.
You may add additional clarifications (e.g., response style), but you must 
keep the above constraints."""


#  DATA MODELS
class PromptRequest(BaseModel):
    question: str


class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: float


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class PromptResponse(BaseModel):
    response: str
    context: list[ContextItem]
    Augmented_prompt: AugmentedPrompt


#  ENDPOINTS

@app.post("/api/prompt", response_model=PromptResponse)
async def prompt_endpoint(request: PromptRequest):
    try:
        # Retrieve
        docs_and_scores = vector_store.similarity_search_with_score(request.question, k=TOP_K)

        formatted_context = ""
        context_list = []

        for doc, score in docs_and_scores:
            formatted_context += f"---\n{doc.page_content}\n"
            context_list.append(ContextItem(
                talk_id=doc.metadata.get("talk_id", "N/A"),
                title=doc.metadata.get("title", "N/A"),
                chunk=doc.page_content,
                score=score
            ))

        # Generate
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEXT + "\n\nCONTEXT:\n{context}"),
            ("user", "{question}")
        ])

        chain = prompt_template | llm

        # We invoke the chain separately to get just the content
        # Note: We reconstruct the system prompt string manually for the return object
        result = chain.invoke({"context": formatted_context, "question": request.question})

        final_sys_prompt = SYSTEM_PROMPT_TEXT + "\n\nCONTEXT:\n" + formatted_context

        return PromptResponse(
            response=result.content,
            context=context_list,
            Augmented_prompt=AugmentedPrompt(System=final_sys_prompt, User=request.question)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats_endpoint():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
