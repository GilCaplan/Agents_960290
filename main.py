import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# -----------------------

from weather_service import WeatherService

# Load Env
load_dotenv()

app = FastAPI()

# 1. Initialize Services
weather_service = WeatherService()

# 2. Initialize Embeddings on GPU (MPS)
print("Loading embeddings on Mac GPU (MPS)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'}
)

# 3. Connect to Pinecone
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

# 4. Initialize LLM
llm = ChatOllama(
    model="llama3",
    temperature=0.2,
    # Ollama automatically uses MPS on Mac, but we can try to force keep-alive to avoid reloading
    keep_alive="5m"
)

# 5. Define the Prompt
prompt_template = """
You are an intelligent agricultural advisor. 
Your goal is to give practical advice using the specific book excerpts provided and the current weather.

Situation:
The user is asking: "{question}"
The current weather is: 
{weather_info}

Book Excerpts (Context):
{context}

Instructions:
1. Analyze the Book Excerpts to find any mention of the user's topic.
2. Look at the Weather (Rain, Wind, Humidity).
3. Combine them: If the books describe a task (like spraying or weeding), explain how the current weather affects it. 
4. If the books suggest "Organic" methods, prioritize those.
5. If the answer is not explicitly in the text, you may infer the best practice based on the weather data provided, but state that this is a general recommendation.

Your Advice:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "weather_info", "question"]
)


# 6. Request Model
class AdviceRequest(BaseModel):
    city: str
    query: str


# 7. API Endpoint
@app.post("/get-advice")
async def get_farming_advice(request: AdviceRequest):
    try:
        # Step A: Get Weather
        weather_info = weather_service.get_current_weather(request.city)

        # Step B: Retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # DEBUG: Print sources
        retrieved_docs = retriever.invoke(request.query)
        print(f"\n--- DEBUG: Retrieved {len(retrieved_docs)} chunks ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"[{i + 1}] Source: {doc.metadata.get('source', 'Unknown')}")
        print("------------------------------------------------\n")

        # Step C: Build Chain
        def format_docs(docs):
            return "\n\n".join(f"[From {d.metadata.get('source', 'Unknown')}]: {d.page_content}" for d in docs)

        rag_chain = (
                {
                    "context": lambda x: format_docs(retrieved_docs),
                    "question": RunnablePassthrough(),
                    "weather_info": lambda x: weather_info
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        # Step D: Run
        print(f"Thinking... (Using Local Llama3 on GPU)")
        result = rag_chain.invoke(request.query)

        return {
            "advice": result,
            "weather_context_used": weather_info
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)