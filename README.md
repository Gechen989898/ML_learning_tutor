# Learning Tutor

Learning Tutor is a textbook-grounded retrieval-augmented generation (RAG) application for asking questions about *Hands-On Machine Learning with Scikit-Learn and TensorFlow*.

The production path uses Azure services end to end: the source PDF is stored in Azure Blob Storage, parsed with Azure Document Intelligence, embedded with Azure OpenAI, indexed in Azure AI Search, and queried through a Streamlit chat interface or CLI. Answers are generated with a grounded LangChain RAG chain that cites retrieved textbook passages and refuses to answer when the provided context is insufficient.

## What It Demonstrates

- Production-style RAG architecture with separate ingestion, indexing, retrieval, and generation layers.
- Layout-aware PDF extraction using Azure Document Intelligence.
- Chunk metadata preservation for chapter, page, source blob, and citation labels.
- Hybrid Azure AI Search retrieval with vector search, keyword search, and optional semantic ranking.
- Conversational query rewriting for follow-up questions.
- Grounded answer generation with citation rules.
- Streamlit UI, CLI query path, Docker runtime, Azure bootstrap scripts, and unit tests.

## Architecture

```text
Azure Blob Storage PDF
        |
        v
Azure Document Intelligence layout extraction
        |
        v
Cleaned page-level LangChain documents
        |
        v
Chapter/page metadata enrichment and chunking
        |
        v
Azure OpenAI embeddings
        |
        v
Azure AI Search index
        |
        v
Query rewrite -> hybrid/semantic retrieval -> grounded answer generation
        |
        v
Streamlit chat UI or CLI response
```

Azure AI Search is the main runtime retrieval backend. FAISS utilities remain in the package for local experimentation and older vector-store workflows.

## Repository Layout

```text
.
├── app/
│   └── streamlit_app.py
├── infra/
│   └── bootstrap.env.example
├── learning_tutor/
│   ├── azure_openai.py
│   ├── azure_search.py
│   ├── data_pipeline.py
│   ├── embedding.py
│   ├── rag_chain.py
│   ├── retrieval_pipeline.py
│   └── services/
│       └── indexing.py
├── scripts/
│   ├── bootstrap_azure.sh
│   ├── build_index.py
│   ├── chat_cli.py
│   └── provision_azure.sh
├── tests/
│   └── test_azure_migration.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Tech Stack

- Python 3.12
- Streamlit
- LangChain
- Azure OpenAI
- Azure AI Search
- Azure Blob Storage
- Azure Document Intelligence
- Azure Identity
- FAISS
- Docker

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file in the repository root. Do not commit real API keys or service keys.

The Streamlit app and CLI require:

```env
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

AZURE_OPENAI_CHAT_ENDPOINT=https://your-chat-resource.openai.azure.com/
AZURE_OPENAI_CHAT_API_KEY=your_chat_key
AZURE_OPENAI_CHAT_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_CHAT_MODEL=your_chat_model_name

AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=your_index_name
AZURE_SEARCH_SEMANTIC_CONFIG_NAME=rag_ml_semantic_config
```

Index building also requires:

```env
AZURE_STORAGE_ACCOUNT_URL=https://yourstorageaccount.blob.core.windows.net
AZURE_STORAGE_ACCOUNT_KEY=your_storage_account_key
AZURE_STORAGE_CONTAINER=documents
AZURE_STORAGE_BLOB_NAME=ml_text_book.pdf
AZURE_BLOB_DOWNLOAD_PATH=azure_data/ml_text_book.pdf

AZURE_DOCUMENT_INTEL_ENDPOINT=https://your-document-intelligence-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTEL_KEY=your_document_intelligence_key
```

Optional embedding controls:

```env
AZURE_OPENAI_EMBEDDING_BATCH_SIZE=16
AZURE_OPENAI_EMBEDDING_RETRY_SECONDS=65
AZURE_OPENAI_EMBEDDING_MAX_RETRIES=8
```

Blob downloads use `AZURE_STORAGE_ACCOUNT_KEY` when set. Without it, the code falls back to `DefaultAzureCredential`, so local development may require:

```bash
az login
```

## Provision Azure Resources

The repo includes scripts for creating the required Azure resources and writing the managed Azure block in `.env`.

Copy the example config if you want to override resource names, SKUs, location, or model deployments:

```bash
cp infra/bootstrap.env.example infra/bootstrap.env
```

Then run:

```bash
bash scripts/bootstrap_azure.sh
```

The bootstrap flow creates or reuses:

1. Resource group.
2. Blob Storage account and container.
3. Azure AI Search service.
4. Azure Document Intelligence account.
5. Azure OpenAI account.
6. Embedding and chat deployments.
7. Azure AI Search index.

The script uploads the configured local PDF, writes Azure settings to `.env`, and runs the index build.

## Build The Search Index

Run the index build after changing the source PDF, chunking logic, embedding deployment, or Azure AI Search schema:

```bash
python scripts/build_index.py
```

The indexing job:

1. Downloads the configured PDF from Azure Blob Storage.
2. Extracts layout-aware text with Azure Document Intelligence.
3. Removes non-content layout roles such as headers, footers, page numbers, and footnotes.
4. Infers chapter and printed page labels.
5. Splits pages into overlapping retrieval chunks.
6. Embeds chunks with Azure OpenAI.
7. Creates or updates the Azure AI Search index.
8. Uploads chunk documents with vectors and citation metadata.

## Run The Streamlit App

```bash
streamlit run app/streamlit_app.py --server.port=8002
```

Open the local URL printed by Streamlit. The app validates Azure OpenAI and Azure AI Search settings at startup, connects to the configured search index, and stores chat history in Streamlit session state.

## Query From The CLI

```bash
python scripts/chat_cli.py
```

The CLI connects to Azure AI Search, accepts one question, and prints the grounded answer.

## Run With Docker

```bash
docker build -t learning-tutor .
docker run --rm -p 8002:8002 --env-file .env learning-tutor
```

Then open:

```text
http://localhost:8002
```

## Retrieval And Generation Behavior

At query time, the Azure-backed flow:

1. Rewrites follow-up questions into standalone search queries when chat history exists.
2. Embeds the standalone query with the configured Azure OpenAI embedding deployment.
3. Searches Azure AI Search with vector and text signals.
4. Uses semantic ranking when enabled.
5. Formats retrieved chunks with exact source labels.
6. Generates an answer using only retrieved context.

If retrieved context is insufficient, the prompt instructs the model to answer:

```text
I don't know based on the provided document.
```

## Example Questions

```text
What is the difference between batch gradient descent and stochastic gradient descent?
```

```text
When would I choose stochastic gradient descent?
```

```text
Why do random forests reduce overfitting compared with a single decision tree?
```

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests
```

The current suite contains 10 tests covering:

- Azure Blob configuration aliases.
- Azure OpenAI configuration parsing for shared and separate chat/embedding resources.
- Document Intelligence layout conversion.
- Chapter and printed page label inference.
- Chunk metadata preservation.

## Development Notes

- `learning_tutor` is the packaged application module.
- `notebook.ipynb` is exploratory and not part of the package.
- `azure_data/` is the default local download path for the Azure Blob source PDF.
- `storage/faiss_index/` is used only for local FAISS vector-store workflows.
- `.env`, local bootstrap overrides, caches, virtual environments, and generated indexes should stay out of source control.
- Keep secrets in local environment variables or deployment configuration, not in committed files.
