import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    print("\n[DEBUG] Loading embeddings model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(transcript: str):
    print("\n[DEBUG] Building vector store...")

    embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(transcript)

    print("[DEBUG] Chunks:", len(chunks))

    docs = [
        Document(page_content=c, metadata={"chunk": i})
        for i, c in enumerate(chunks)
    ]

    # IMPORTANT: DO NOT reuse old DB
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print("[DEBUG] Vector store created successfully")
    return vector_store


def load_vector_store() -> Chroma:
    print("\n[DEBUG] Loading existing vector DB...")

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("[DEBUG] Total docs in DB:", vector_store._collection.count())

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 6):
    print(f"\n[DEBUG] Creating retriever with k={k}")

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )