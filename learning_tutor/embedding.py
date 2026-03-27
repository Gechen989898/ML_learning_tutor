"""Embedding and vector store utilities for semantic retrieval."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(model=DEFAULT_EMBEDDING_MODEL):
    """Create the embedding client used across indexing and retrieval.

    Using one embedding model for both document chunks and queries ensures they
    share the same vector space, which is required for meaningful similarity
    search.

    Args:
        model: Name of the OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: Configured embedding client.
    """
    return OpenAIEmbeddings(model=model)


def embedding_vector(chunks, model=DEFAULT_EMBEDDING_MODEL):
    """Build a FAISS vector store from chunk documents.

    Embeddings provide the semantic retrieval layer for the RAG system. FAISS is
    used because it offers fast local nearest-neighbor search with minimal
    operational overhead for a small to medium document collection.

    Args:
        chunks: Chunk documents produced by the data pipeline.
        model: Name of the OpenAI embedding model.

    Returns:
        FAISS: In-memory vector store containing embedded chunks.

    Notes:
        Smaller embedding models reduce cost and latency, but may lose some
        semantic precision compared with larger models.
    """
    embeddings = get_embeddings(model=model)
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


def save_vector_store(vector_store, index_dir):
    """Persist a FAISS index to disk.

    Persistence avoids recomputing embeddings on every application startup,
    which keeps deployment latency and API cost manageable.

    Args:
        vector_store: FAISS vector store to save.
        index_dir: Output directory for the serialized index files.

    Returns:
        None
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))


def load_vector_store(index_dir, model=DEFAULT_EMBEDDING_MODEL):
    """Load a persisted FAISS index if one exists.

    Args:
        index_dir: Directory containing the serialized FAISS index.
        model: Name of the OpenAI embedding model used for query encoding.

    Returns:
        FAISS | None: Loaded vector store, or ``None`` if the index is missing.

    Notes:
        The same embedding model should be used at load time and index build
        time so query vectors remain comparable to stored document vectors.
    """
    index_path = Path(index_dir)
    if not index_path.exists():
        return None

    embeddings = get_embeddings(model=model)
    return FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
