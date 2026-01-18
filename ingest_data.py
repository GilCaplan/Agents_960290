import os
import json
import glob
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def process_book_json(filepath):
    """
    Reads your specific agriculture JSON format and extracts text.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []

    def extract_text(node, current_path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{current_path} > {key}" if current_path else key
                extract_text(value, new_path)
        elif isinstance(node, str):
            content = f"Source: {os.path.basename(filepath)}\nSection: {current_path}\nContent: {node}"
            documents.append(content)

    extract_text(data)
    return documents


def main():
    # 1. Initialize Pinecone Client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Check and Reset Index
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' exists but might have the wrong dimensions.")
        print(f"DELETING '{INDEX_NAME}' to ensure a clean start...")
        pc.delete_index(INDEX_NAME)
        # Wait a moment for deletion to register
        time.sleep(5)

    print(f"Creating new index: {INDEX_NAME} with 384 dimensions...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Correct size for 'all-MiniLM-L6-v2'
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    # 3. Initialize Embeddings (Local/Free)
    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Load and Chunk Data
    all_texts = []
    # We purposefully look for 'source_*.json' to avoid ingesting the weather files
    files = glob.glob("data/source_*.json")

    print(f"Found {len(files)} source files...")

    for file in files:
        print(f"Processing {file}...")
        raw_docs = process_book_json(file)
        all_texts.extend(raw_docs)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents(all_texts)
    print(f"Generated {len(chunks)} chunks.")

    # 5. Upload to Pinecone
    print("Uploading to Pinecone...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print("Ingestion complete!")


if __name__ == "__main__":
    main()