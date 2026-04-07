"""Azure Blob Storage and Azure AI Search helpers for the RAG pipeline."""

import os
import re
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from langchain_core.documents import Document


DEFAULT_VECTOR_FIELD = "content_vector"
DEFAULT_SEMANTIC_CONFIG = "rag_ml_semantic_config"
DEFAULT_AZURE_DOWNLOAD_PATH = "azure_data/ml_text_book.pdf"


def _get_env(name, default=None):
    """Read an environment variable, tolerating whitespace around keys."""
    value = os.getenv(name)
    if value is not None:
        return value.strip()

    for key, candidate in os.environ.items():
        if key.strip() == name:
            return candidate.strip()
    return default


def get_search_config():
    """Return Azure AI Search configuration from environment variables."""
    endpoint = _get_env("AZURE_SEARCH_ENDPOINT")
    api_key = _get_env("AZURE_SEARCH_API_KEY")
    index_name = _get_env("AZURE_SEARCH_INDEX_NAME")
    semantic_config_name = _get_env(
        "AZURE_SEARCH_SEMANTIC_CONFIG_NAME",
        DEFAULT_SEMANTIC_CONFIG,
    )

    missing = [
        name
        for name, value in {
            "AZURE_SEARCH_ENDPOINT": endpoint,
            "AZURE_SEARCH_API_KEY": api_key,
            "AZURE_SEARCH_INDEX_NAME": index_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Azure AI Search environment variables: {missing}")

    return endpoint, api_key, index_name, semantic_config_name


def get_blob_config():
    """Return Azure Blob Storage configuration from environment variables."""
    account_url = _get_env("AZURE_STORAGE_ACCOUNT_URL") or _get_env(
        "AZURE_ACCOUNT_ENDPOINT"
    )
    container_name = _get_env("AZURE_STORAGE_CONTAINER", "documents")
    blob_name = _get_env("AZURE_STORAGE_BLOB_NAME")

    if not account_url:
        raise ValueError(
            "Missing Azure Blob Storage account URL. Set "
            "`AZURE_STORAGE_ACCOUNT_URL` or `AZURE_ACCOUNT_ENDPOINT`."
        )

    return account_url, container_name, blob_name


def get_search_client(index_name=None):
    """Create an Azure AI Search document client."""
    endpoint, api_key, default_index_name, _ = get_search_config()
    return SearchClient(
        endpoint=endpoint,
        index_name=index_name or default_index_name,
        credential=AzureKeyCredential(api_key),
    )


def get_index_client():
    """Create an Azure AI Search index management client."""
    endpoint, api_key, _, _ = get_search_config()
    return SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )


def get_blob_service_client():
    """Create a Blob Storage client using the local/default Azure identity."""
    account_url, _, _ = get_blob_config()
    return BlobServiceClient(
        account_url=account_url,
        credential=DefaultAzureCredential(),
    )


def download_blob_to_file(
    local_path=DEFAULT_AZURE_DOWNLOAD_PATH,
    container_name=None,
    blob_name=None,
):
    """Download the configured source blob to a local file path."""
    _, default_container_name, default_blob_name = get_blob_config()
    container_name = container_name or default_container_name
    blob_name = blob_name or default_blob_name

    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    if not blob_name:
        blob_name = next(container_client.list_blobs()).name

    download_path = Path(local_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    with download_path.open("wb") as download_file:
        download_file.write(container_client.download_blob(blob_name).readall())

    return str(download_path), blob_name


def create_or_update_search_index(
    vector_dimensions,
    index_name=None,
    semantic_config_name=None,
):
    """Create or update the Azure AI Search index used by this app."""
    _, _, default_index_name, default_semantic_config = get_search_config()
    index_name = index_name or default_index_name
    semantic_config_name = semantic_config_name or default_semantic_config

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="chapter", type=SearchFieldDataType.String),
        SimpleField(name="page", type=SearchFieldDataType.Int32),
        SearchableField(
            name="metadata_label",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="source_blob",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name=DEFAULT_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name="vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[
                        SemanticField(field_name="chapter"),
                        SemanticField(field_name="metadata_label"),
                    ],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    get_index_client().create_or_update_index(index)
    return index_name


def _make_document_id(source_blob, index):
    """Create a stable Azure Search key for a chunk."""
    safe_source = re.sub(r"[^A-Za-z0-9_\-=]+", "-", source_blob).strip("-")
    return f"{safe_source}-{index:06d}"


def chunks_to_search_documents(chunks, embeddings, source_blob):
    """Convert LangChain chunks into Azure AI Search upload documents."""
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    documents = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        documents.append(
            {
                "id": _make_document_id(source_blob, index),
                "content": chunk.page_content,
                "chapter": chunk.metadata.get("chapter", ""),
                "page": int(chunk.metadata.get("page", 0)),
                "metadata_label": chunk.metadata.get("metadata_label", ""),
                "source_blob": source_blob,
                "content_vector": vector,
            }
        )
    return documents


def upload_documents(documents, batch_size=100, search_client=None):
    """Upload Azure AI Search documents in batches."""
    search_client = search_client or get_search_client()
    uploaded = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        results = search_client.upload_documents(documents=batch)
        uploaded += sum(1 for result in results if result.succeeded)
    return uploaded


def search_hybrid_semantic(
    query,
    embeddings,
    search_client=None,
    k_nearest_neighbors=20,
    top=5,
    use_semantic=True,
    semantic_config_name=None,
):
    """Run Azure AI Search hybrid vector/text retrieval and return Documents."""
    search_client = search_client or get_search_client()
    _, _, _, default_semantic_config = get_search_config()
    semantic_config_name = semantic_config_name or default_semantic_config
    query_vector = embeddings.embed_query(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k_nearest_neighbors,
        fields=DEFAULT_VECTOR_FIELD,
    )
    search_kwargs = {
        "search_text": query,
        "vector_queries": [vector_query],
        "select": ["content", "chapter", "page", "metadata_label", "source_blob"],
        "top": top,
    }
    if use_semantic:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = semantic_config_name

    results = search_client.search(**search_kwargs)
    return [
        Document(
            page_content=result.get("content", ""),
            metadata={
                "chapter": result.get("chapter", ""),
                "page": result.get("page", 0),
                "metadata_label": result.get("metadata_label", "Unknown source"),
                "source_blob": result.get("source_blob", ""),
            },
        )
        for result in results
    ]
