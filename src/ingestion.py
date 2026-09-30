import os
import sys
import time
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Ensure parent directory is in path when executed directly
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.config import (
    DEFAULT_PDF_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_DIMENSION,
    VECTOR_METRIC,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def get_embeddings():
    """Initialize the open-source Ollama embedding model."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def initialize_pinecone_index(
    api_key: str,
    index_name: str,
    dimension: int = VECTOR_DIMENSION,
    metric: str = VECTOR_METRIC,
    cloud: str = PINECONE_CLOUD,
    region: str = PINECONE_REGION,
) -> Pinecone:
    """Ensure Pinecone index exists with the correct dimension and metric."""
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Please add your Pinecone API Key to the .env file."
        )

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"[Pinecone] Creating index '{index_name}' (dimension={dimension}, metric={metric})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for index readiness
        while not pc.describe_index(index_name).status["ready"]:
            print("[Pinecone] Waiting for index to be ready...")
            time.sleep(2)
        print(f"[Pinecone] Index '{index_name}' is ready.")
    else:
        index_meta = pc.describe_index(index_name)
        if index_meta.dimension != dimension:
            print(
                f"[Pinecone Warning] Index '{index_name}' has dimension {index_meta.dimension}, "
                f"but embedding model '{EMBEDDING_MODEL}' uses dimension {dimension}."
            )
        print(f"[Pinecone] Index '{index_name}' already exists and is ready.")

    return pc


def run_ingestion(pdf_path: str = DEFAULT_PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """
    Ingestion pipeline:
    1. Load PDF document
    2. Split into chunks
    3. Initialize Pinecone index
    4. Embed & upsert chunks into Pinecone vector store
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF source document not found at: {pdf_path}")

    print(f"[Ingestion] Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[Ingestion] Loaded {len(docs)} pages.")

    print(f"[Ingestion] Splitting text with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[Ingestion] Generated {len(chunks)} text chunks.")

    # Initialize Pinecone
    initialize_pinecone_index(
        api_key=PINECONE_API_KEY,
        index_name=index_name,
        dimension=VECTOR_DIMENSION,
    )

    # Open-source embeddings
    print(f"[Ingestion] Initializing open-source embedding model: '{EMBEDDING_MODEL}' via Ollama...")
    embeddings = get_embeddings()

    # Index into Pinecone
    print(f"[Ingestion] Generating embeddings and upserting chunks into Pinecone index '{index_name}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("[Ingestion] Ingestion completed successfully!")
    return vector_store


if __name__ == "__main__":
    run_ingestion()
