# Learning Tutor

Learning Tutor is a textbook-backed retrieval-augmented generation (RAG) app for asking grounded questions about *Hands-On Machine Learning with Scikit-Learn and TensorFlow*.

The project loads a source PDF, prepares clean retrieval chunks, embeds them with OpenAI embeddings, indexes them in Azure AI Search, and serves answers through a Streamlit chat UI. It also includes local FAISS helpers for development workflows.

## Why It Is Useful

- Answers are grounded in retrieved textbook passages instead of free-form model knowledge.
- Follow-up questions are rewritten into standalone retrieval queries so conversational context is easier to search.
- Azure AI Search combines text, vector, and optional semantic retrieval for production use.
- Source labels are preserved and shown in the Streamlit UI so users can inspect answer provenance.
- Indexing is separated from app startup, which keeps the web process focused on serving queries.
- Docker support makes the Streamlit app easier to run in a consistent Python 3.12 environment.

## How It Works

1. [`scripts/build_index.py`](scripts/build_index.py) downloads the configured PDF from Azure Blob Storage.
2. [`learning_tutor/data_pipeline.py`](learning_tutor/data_pipeline.py) loads pages, attaches chapter/page metadata, cleans text, and splits pages into chunks.
3. [`learning_tutor/embedding.py`](learning_tutor/embedding.py) creates OpenAI embeddings with `text-embedding-3-small`.
4. [`learning_tutor/azure_search.py`](learning_tutor/azure_search.py) creates or updates an Azure AI Search index with vector and semantic search configuration.
5. [`learning_tutor/retrieval_pipeline.py`](learning_tutor/retrieval_pipeline.py) rewrites conversational questions and retrieves ranked context.
6. [`learning_tutor/rag_chain.py`](learning_tutor/rag_chain.py) builds the LangChain answer chain and applies citation-focused prompt rules.
7. [`app/streamlit_app.py`](app/streamlit_app.py) runs the chat UI and displays retrieved source labels.

## Tech Stack

- [Python 3.12](https://www.python.org/)
- [Streamlit](https://streamlit.io/) for the web chat UI
- [LangChain](https://docs.langchain.com/) for prompts, runnables, document objects, and OpenAI integrations
- [OpenAI](https://platform.openai.com/docs/) for chat completions and embeddings
- [`text-embedding-3-small`](https://platform.openai.com/docs/models/text-embedding-3-small) as the default embedding model
- [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) for production retrieval
- [Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/) for the source PDF
- [Azure Identity](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme) for Blob Storage authentication
- [FAISS](https://github.com/facebookresearch/faiss) for local vector store support
- [pypdf](https://pypdf.readthedocs.io/) through LangChain's PDF loader
- [python-dotenv](https://pypi.org/project/python-dotenv/) for local `.env` loading
- [Docker](https://docs.docker.com/) for containerized app runtime

## Project Structure

```text
.
├── app/
├── azure_data/
├── data/
├── learning_tutor/
│   └── services/
├── scripts/
├── src/
├── storage/
│   └── faiss_index/
├── tests/
├── .dockerignore
├── Dockerfile
├── notebook.ipynb
├── pyproject.toml
├── README.md
└── requirements.txt
```

[`app/`](app/) contains the Streamlit application.

[`learning_tutor/`](learning_tutor/) contains the package used by the app and scripts.

[`learning_tutor/services/`](learning_tutor/services/) contains index lifecycle helpers.

[`scripts/`](scripts/) contains CLI entrypoints for indexing and one-off chat queries.

[`data/`](data/) contains the local textbook PDF.

[`azure_data/`](azure_data/) is the default download target for Blob Storage indexing.

[`storage/faiss_index/`](storage/faiss_index/) is the local FAISS persistence path and is excluded from the Docker build context.

[`src/`](src/) and [`tests/`](tests/) are present as development directories, but the packaged module is declared in [`pyproject.toml`](pyproject.toml) as `learning_tutor`.

## Getting Started

### Prerequisites

- Python 3.12
- An OpenAI API key
- An Azure AI Search service and index name
- An Azure Storage account with a PDF in Blob Storage if you want to build the Azure index from source
- Azure credentials available through `DefaultAzureCredential`, such as `az login` for local development

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Configure

Create a `.env` file in the repository root.

```env
OPENAI_API_KEY=your_openai_api_key
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_azure_search_admin_or_query_key
AZURE_SEARCH_INDEX_NAME=your_index_name

AZURE_STORAGE_ACCOUNT_URL=https://yourstorageaccount.blob.core.windows.net
AZURE_STORAGE_CONTAINER=documents
AZURE_STORAGE_BLOB_NAME=Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf

AZURE_SEARCH_SEMANTIC_CONFIG_NAME=rag_ml_semantic_config
AZURE_BLOB_DOWNLOAD_PATH=azure_data/ml_text_book.pdf
```

`AZURE_SEARCH_SEMANTIC_CONFIG_NAME` and `AZURE_BLOB_DOWNLOAD_PATH` are optional. The app uses defaults when they are not set.

### Build The Azure Search Index

Run this when the source PDF, chunking logic, or embedding model changes.

```bash
python scripts/build_index.py
```

### Run The Streamlit App

```bash
streamlit run app/streamlit_app.py --server.port=8002
```

Open the local URL printed by Streamlit and ask a question about the textbook.

### Run A CLI Query

```bash
python scripts/chat_cli.py
```

### Run With Docker

```bash
docker build -t learning-tutor .
docker run --rm -p 8002:8002 --env-file .env learning-tutor
```

Then open `http://localhost:8002`.

## Usage Example

Ask questions that can be answered from the indexed textbook:

```text
What is the difference between batch gradient descent and stochastic gradient descent?
```

Follow-up questions can rely on the previous turn:

```text
When would I choose the second one?
```

The retrieval pipeline rewrites the follow-up into a standalone query before searching.

## Development Notes

- The Streamlit app checks for `OPENAI_API_KEY`, `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, and `AZURE_SEARCH_INDEX_NAME` before startup.
- [`Dockerfile`](Dockerfile) exposes port `8002` and runs Streamlit on `0.0.0.0`.
- [`.dockerignore`](.dockerignore) excludes `.env`, Python caches, notebook checkpoints, and `storage/faiss_index/`.
- The local FAISS helper path is available in [`learning_tutor/services/indexing.py`](learning_tutor/services/indexing.py), but the current app entrypoint uses Azure AI Search.
- No custom web fonts are referenced by the current code.

## Getting Help

- Start with the code paths listed in [How It Works](#how-it-works).
- For Streamlit UI behavior, see the [Streamlit docs](https://docs.streamlit.io/).
- For Azure Search indexing and vector search, see the [Azure AI Search docs](https://learn.microsoft.com/en-us/azure/search/).
- For LangChain chain composition, see the [LangChain Python docs](https://docs.langchain.com/).
- If this project is published on GitHub, open an issue with the failing command, expected behavior, actual behavior, and relevant environment details.

## Maintainers And Contributing

This repository is currently maintained by the project owner.

Contributions should stay focused on the RAG pipeline, indexing flow, deployment setup, and developer experience. Before opening a pull request:

1. Keep changes small and scoped.
2. Update this README when setup, dependencies, ports, or environment variables change.
3. Avoid committing secrets, generated caches, local vector indexes, or notebook checkpoints.
4. Add or update tests when behavior changes.
5. Run the relevant app, CLI, or indexing command locally before asking for review.

There is no separate `CONTRIBUTING.md` or `LICENSE` file in the repository yet. Add those files before publishing the project for broader reuse.
