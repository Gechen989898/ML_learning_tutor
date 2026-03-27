"""Streamlit UI for the textbook-backed RAG tutor."""

import os

import streamlit as st
from dotenv import load_dotenv

from learning_tutor.rag_chain import build_rag_chain
from learning_tutor.retrieval_pipeline import multi_stage_retrieval
from learning_tutor.services.indexing import load_or_build_vector_store


load_dotenv()

DEFAULT_TITLE = "Your learning tutor is here ! "
DEFAULT_GREETING = "Ask me anything about the book."
INDEX_STATUS_KEY = "index_status"
DEFAULT_FILE_PATH = os.getenv(
    "PDF_SOURCE_PATH",
    "data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf",
)
DEFAULT_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "storage/faiss_index")


def has_openai_api_key():
    """Check whether the application can call OpenAI services.

    Args:
        None

    Returns:
        bool: ``True`` when the API key is available in the environment.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner=False)
def get_vector_store(file_path, index_dir):
    """Load or build the cached vector store for the web app.

    Args:
        file_path: Path to the source PDF.
        index_dir: Directory where the FAISS index is stored.

    Returns:
        tuple: Pair of ``(vector_store, status_message)``.
    """
    return load_or_build_vector_store(file_path, index_dir)


@st.cache_resource(show_spinner=False)
def get_chain(file_path, index_dir):
    """Build the cached RAG chain for the current app configuration.

    Args:
        file_path: Path to the source PDF.
        index_dir: Directory where the FAISS index is stored.

    Returns:
        Runnable: Answer-generation chain backed by retrieval.
    """
    vector_store, _ = get_vector_store(file_path, index_dir)
    return build_rag_chain(vector_store)


def format_sources(docs):
    """Extract unique source labels for UI display.

    Args:
        docs: Retrieved documents returned by the retrieval pipeline.

    Returns:
        list[str]: Deduplicated source labels in original order.
    """
    labels = []
    for doc in docs:
        label = doc.metadata.get("metadata_label", "Unknown source")
        if label not in labels:
            labels.append(label)
    return labels


def get_chat_history(messages):
    """Filter session messages down to conversational turns.

    Args:
        messages: Full Streamlit session message list.

    Returns:
        list[dict]: User and assistant messages only.
    """
    return [message for message in messages if message["role"] in {"user", "assistant"}]


st.set_page_config(page_title=DEFAULT_TITLE, layout="wide")
st.title(DEFAULT_TITLE)
st.caption("Chat with your book-backed RAG pipeline.")

file_path = DEFAULT_FILE_PATH
index_dir = DEFAULT_INDEX_DIR

if not has_openai_api_key():
    st.error(
        "Missing `OPENAI_API_KEY`. Set it in your environment or `.env` file before starting the app."
    )
    st.stop()

if INDEX_STATUS_KEY not in st.session_state:
    with st.spinner("Preparing the knowledge base..."):
        _, index_status = get_vector_store(file_path, index_dir)
    st.session_state[INDEX_STATUS_KEY] = index_status

st.info(st.session_state[INDEX_STATUS_KEY])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": DEFAULT_GREETING, "sources": []}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

user_query = st.chat_input("Ask a question about the book")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query, "sources": []}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vector_store, _ = get_vector_store(file_path, index_dir)
            chain = get_chain(file_path, index_dir)
            chat_history = get_chat_history(st.session_state.messages[:-1])
            # Sources are fetched explicitly for display so the UI can expose
            # provenance independent of the final answer wording.
            docs = multi_stage_retrieval(
                user_query,
                vector_store=vector_store,
                chat_history=chat_history,
            )
            response = chain.invoke(
                {
                    "question": user_query,
                    "chat_history": chat_history,
                }
            )
            sources = format_sources(docs)

        st.markdown(response)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- `{source}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
