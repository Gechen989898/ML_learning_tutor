"""CLI entrypoint for querying the textbook-backed RAG system."""

import os

from dotenv import load_dotenv

from learning_tutor.rag_chain import build_rag_chain
from learning_tutor.services.indexing import load_or_build_vector_store


load_dotenv()

DEFAULT_FILE_PATH = os.getenv(
    "PDF_SOURCE_PATH",
    "data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf",
)
DEFAULT_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "storage/faiss_index")


if __name__ == "__main__":
    vector_store, status = load_or_build_vector_store(DEFAULT_FILE_PATH, DEFAULT_INDEX_DIR)
    chain = build_rag_chain(vector_store)
    print(status)
    print("Ask a question about the book.")
    user_query = input("Question: ")
    response = chain.invoke({"question": user_query, "chat_history": []})
    print(response)
