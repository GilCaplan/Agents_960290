import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load keys from .env file
load_dotenv()

# --- CONFIGURATION ---
TEST_MODE = False  # Set to False only when ready for full upload!
TEST_LIMIT = 150  # Number of talks to process in test mode
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
INDEX_NAME = "ted-rag-index"

# Custom Model Config
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"


def process_data():
    print(f"🚀 Starting ingestion. Test Mode: {TEST_MODE}")

    # 1. Load Data
    if not os.path.exists("ted_talks_en.csv"):
        print("❌ Error: ted_talks_en.csv not found.")
        return

    df = pd.read_csv("ted_talks_en.csv")

    if TEST_MODE:
        df = df.head(TEST_LIMIT)
        print(f"⚠️ TEST MODE: Only processing first {TEST_LIMIT} rows.")

    documents = []

    # 2. Prepare Text with Metadata
    for index, row in df.iterrows():
        # Combine relevant fields for the model to "read"
        text_content = (
            f"Title: {row['title']}\n"
            f"Speaker: {row['speaker_1']}\n"
            f"Topics: {row['topics']}\n"
            f"Description: {row['description']}\n"
            f"Transcript: {row['transcript']}"
        )

        metadata = {
            "talk_id": str(row['talk_id']),
            "title": row['title'],
            "speaker": row['speaker_1'],
            "url": row['url']
        }

        documents.append({"text": text_content, "metadata": metadata})

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    final_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            final_docs.append(
                {"page_content": chunk, "metadata": doc["metadata"]}
            )

    print(f"🧩 Created {len(final_docs)} chunks.")

    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"📦 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(5)

        # 5. Upload to Pinecone
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ.get("LLMOD_API_KEY"),
        base_url=LLMOD_BASE_URL
    )

    from langchain_core.documents import Document
    lc_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in final_docs]

    print("📤 Uploading vectors...")
    PineconeVectorStore.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("🎉 Ingestion Complete!")


if __name__ == "__main__":
    process_data()