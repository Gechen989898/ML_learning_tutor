from pathlib import Path

from learning_tutor.data_pipeline import load_data, split_chunk, split_clean_chunks
from learning_tutor.embedding import embedding_vector, load_vector_store, save_vector_store


def build_vector_store(file_path):
    document = load_data(file_path)
    filtered_docs = split_chunk(document)
    chunks = split_clean_chunks(filtered_docs)
    return embedding_vector(chunks)


def load_or_build_vector_store(file_path, index_dir):
    index_path = Path(index_dir)
    vector_store = load_vector_store(index_dir)
    if vector_store is not None:
        return vector_store, f"Loaded existing FAISS index from `{index_path}`."

    vector_store = build_vector_store(file_path)
    save_vector_store(vector_store, index_dir)
    return vector_store, f"Built a new FAISS index and saved it to `{index_path}`."
