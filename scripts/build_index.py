"""CLI entrypoint for building the Azure AI Search index."""

import os

from dotenv import load_dotenv

from learning_tutor.services.indexing import build_azure_search_index_from_blob


load_dotenv()

DEFAULT_AZURE_DOWNLOAD_PATH = os.getenv(
    "AZURE_BLOB_DOWNLOAD_PATH",
    "azure_data/ml_text_book.pdf",
)


if __name__ == "__main__":
    status = build_azure_search_index_from_blob(local_path=DEFAULT_AZURE_DOWNLOAD_PATH)
    print(status)
