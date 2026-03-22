from dotenv import load_dotenv
load_dotenv()

from src.data_pipeline import load_data, split_chunk, split_clean_chunks
from src.embedding import embedding_vector, load_vector_store, save_vector_store
from src.rag_chain import build_rag_chain


DEFAULT_FILE_PATH = "data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf"
DEFAULT_INDEX_DIR = "storage/faiss_index"


def build_vector_store(file_path):
    document = load_data(file_path)
    filtered_docs = split_chunk(document)
    chunks = split_clean_chunks(filtered_docs)
    return embedding_vector(chunks)


def load_or_build_vector_store(file_path, index_dir):
    vector_store = load_vector_store(index_dir)
    if vector_store is not None:
        print(f"Loaded FAISS index from {index_dir}")
        return vector_store

    print("No saved FAISS index found. Building a new one...")
    vector_store = build_vector_store(file_path)
    save_vector_store(vector_store, index_dir)
    print(f"Saved FAISS index to {index_dir}")
    return vector_store


if __name__ == "__main__":
    vector_store = load_or_build_vector_store(DEFAULT_FILE_PATH, DEFAULT_INDEX_DIR)
    chain = build_rag_chain(vector_store)
    print("please ask me anything about this book?")
    user_query = input("Question : ")
    response = chain.invoke(user_query)
    print(response)
