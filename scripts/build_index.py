from dotenv import load_dotenv

from learning_tutor.services.indexing import load_or_build_vector_store

import os


load_dotenv()

DEFAULT_FILE_PATH = os.getenv(
    "PDF_SOURCE_PATH",
    "data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf",
)
DEFAULT_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "storage/faiss_index")


if __name__ == "__main__":
    _, status = load_or_build_vector_store(DEFAULT_FILE_PATH, DEFAULT_INDEX_DIR)
    print(status)
