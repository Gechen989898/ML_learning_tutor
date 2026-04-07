"""Index lifecycle helpers for the retrieval pipeline."""

from pathlib import Path

from learning_tutor.azure_search import (
    chunks_to_search_documents,
    create_or_update_search_index,
    download_blob_to_file,
    upload_documents,
)
from learning_tutor.data_pipeline import load_data, split_chunk, split_clean_chunks
from learning_tutor.embedding import (
    embedding_vector,
    get_embeddings,
    load_vector_store,
    save_vector_store,
)


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


def build_azure_search_index_from_blob(local_path=None):
    """Build the Azure AI Search index from the configured Blob Storage PDF.

    This is the production indexing path: the app should query Azure AI Search
    at runtime, while this function is run separately when the source document
    or chunking logic changes.

    Args:
        local_path: Optional local download path for the source blob.

    Returns:
        str: Status message describing the completed indexing operation.
    """
    local_path, source_blob = download_blob_to_file(local_path=local_path)
    documents = load_data(local_path)
    filtered_docs = split_chunk(documents)
    chunks = split_clean_chunks(filtered_docs)

    embeddings = get_embeddings()
    vector_dimensions = len(embeddings.embed_query(chunks[0].page_content))
    index_name = create_or_update_search_index(vector_dimensions=vector_dimensions)
    search_documents = chunks_to_search_documents(
        chunks=chunks,
        embeddings=embeddings,
        source_blob=source_blob,
    )
    uploaded_count = upload_documents(search_documents)
    return (
        f"Indexed `{source_blob}` into Azure AI Search index `{index_name}` "
        f"with {uploaded_count} chunks."
    )
