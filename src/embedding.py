from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(model=DEFAULT_EMBEDDING_MODEL):
    return OpenAIEmbeddings(model=model)


def embedding_vector(chunks, model=DEFAULT_EMBEDDING_MODEL):
    embeddings = get_embeddings(model=model)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


def save_vector_store(vector_store, index_dir):
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))


def load_vector_store(index_dir, model=DEFAULT_EMBEDDING_MODEL):
    index_path = Path(index_dir)
    if not index_path.exists():
        return None

    embeddings = get_embeddings(model=model)
    return FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
