# Learning Tutor

An OpenAI-powered RAG tutor for *Hands-On Machine Learning with Scikit-Learn and TensorFlow*.

This project ingests a textbook PDF, builds a FAISS vector index, rewrites follow-up questions into standalone queries, reranks candidate passages with an LLM, and serves the experience through a Streamlit chat interface.

## Live Demo

Try the deployed app on Azure:

https://rag-streamlit-123456.azurewebsites.net/

## What It Does

- Answers questions grounded in the source document only
- Supports conversational follow-up questions
- Rewrites ambiguous user queries using chat history
- Retrieves candidate chunks from a FAISS vector store
- Reranks results with an LLM before generating the final answer
- Displays source labels so answers stay traceable

## Demo Flow

1. Load the textbook PDF
2. Split pages into clean chunks with chapter-aware metadata
3. Embed chunks with OpenAI embeddings
4. Store vectors in a local FAISS index
5. Retrieve, rerank, and answer inside the Streamlit app

## Tech Stack

- Python 3.12
- Streamlit
- LangChain
- OpenAI
- FAISS
- PyPDF
- Docker

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── learning_tutor/
│   ├── data_pipeline.py          # PDF loading, cleaning, chunking
│   ├── embedding.py              # Embeddings + FAISS persistence
│   ├── rag_chain.py              # Prompting and answer generation
│   ├── retrieval_pipeline.py     # Query rewrite, retrieval, reranking
│   └── services/
│       └── indexing.py           # Load/build vector store orchestration
├── scripts/
│   ├── build_index.py            # Prebuild the FAISS index
│   └── chat_cli.py               # Terminal chat entrypoint
├── data/                         # Source PDF
├── storage/                      # Local FAISS index output
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
PDF_SOURCE_PATH=data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf
FAISS_INDEX_DIR=storage/faiss_index
```

### 3. Build the vector index

```bash
python scripts/build_index.py
```

### 4. Start the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser.

## CLI Mode

You can also chat with the tutor in the terminal:

```bash
python scripts/chat_cli.py
```

## Docker

Build the image:

```bash
docker build -t learning-tutor .
```

Run the container:

```bash
docker run --rm -p 8001:8001 --env-file .env learning-tutor
```

Then open `http://localhost:8001`.

## How Retrieval Works

The retrieval pipeline is multi-stage:

1. Rewrite the latest user question into a standalone form when chat history exists
2. Run similarity search against the FAISS vector store
3. Ask an LLM to rerank the retrieved candidates
4. Feed the top sources into the final answer chain

This helps the app handle vague follow-ups better than plain vector search alone.

## Current Deployment Shape

- The Streamlit app loads environment variables with `python-dotenv`
- OpenAI credentials are read from `OPENAI_API_KEY`
- The app can load an existing FAISS index or build one if missing
- Docker is available for containerized deployment

For production deployment, prebuilding the FAISS index before startup is the safer approach.

## Notes

- `.env` is intended for local development and is ignored by Docker build context
- `storage/faiss_index/` is excluded from Docker build context in the current setup
- The app stops early if `OPENAI_API_KEY` is missing

## Roadmap Ideas

- Separate production and development dependencies
- Prebuild and persist the FAISS index for cloud deployment
- Add CI/CD for image build and deployment
- Add health checks and production configuration for Azure

## License

Add a license file if you plan to distribute or publish the project.
