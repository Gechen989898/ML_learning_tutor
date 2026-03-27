"""Index lifecycle helpers for the retrieval pipeline."""

from pathlib import Path

from learning_tutor.data_pipeline import load_data, split_chunk, split_clean_chunks
from learning_tutor.embedding import embedding_vector, load_vector_store, save_vector_store


def build_vector_store(file_path):
    """Build a new vector store from the source PDF.

    This function ties together document loading, chunk preparation, and
    embedding so the rest of the application can work with a single retrieval
    artifact.

    Args:
        file_path: Path to the source PDF.

    Returns:
        FAISS: Newly built vector store.
    """
    document = load_data(file_path)
    filtered_docs = split_chunk(document)
    chunks = split_clean_chunks(filtered_docs)
    return embedding_vector(chunks)


def load_or_build_vector_store(file_path, index_dir):
    """Load an existing vector store or build one on demand.

    This keeps startup fast in the common case while still allowing a clean
    environment to bootstrap itself automatically.

    Args:
        file_path: Path to the source PDF.
        index_dir: Directory where the FAISS index is stored.

    Returns:
        tuple: Pair of ``(vector_store, status_message)``.
    """
    index_path = Path(index_dir)
    vector_store = load_vector_store(index_dir)
    if vector_store is not None:
        return vector_store, f"Loaded existing FAISS index from `{index_path}`."

    # Embedding is the most expensive preprocessing step, so the built index is
    # persisted and reused across later runs.
    vector_store = build_vector_store(file_path)
    save_vector_store(vector_store, index_dir)
    return vector_store, f"Built a new FAISS index and saved it to `{index_path}`."
