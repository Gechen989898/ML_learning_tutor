"""Streamlit UI for the textbook-backed RAG tutor."""

import os

import streamlit as st
from dotenv import load_dotenv

from learning_tutor.azure_search import get_search_client
from learning_tutor.rag_chain import build_azure_rag_chain
from learning_tutor.retrieval_pipeline import multi_stage_azure_retrieval


load_dotenv()

DEFAULT_TITLE = "Your learning tutor is here ! "
DEFAULT_GREETING = "Ask me anything about the book."
INDEX_STATUS_KEY = "index_status"


def has_required_settings():
    """Check whether the application has required cloud settings.

    Args:
        None

    Returns:
        bool: ``True`` when required environment variables are available.
    """
    required_settings = [
        "OPENAI_API_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    return all(os.getenv(setting) for setting in required_settings)


@st.cache_resource(show_spinner=False)
def get_cached_search_client():
    """Create the cached Azure AI Search client for the web app.

    Args:
        None

    Returns:
        SearchClient: Azure AI Search document client.
    """
    return get_search_client()


@st.cache_resource(show_spinner=False)
def get_chain():
    """Build the cached RAG chain for the current app configuration.

    Args:
        None

    Returns:
        Runnable: Answer-generation chain backed by retrieval.
    """
    return build_azure_rag_chain(get_cached_search_client())


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

if not has_required_settings():
    st.error(
        "Missing required settings. Set `OPENAI_API_KEY`, `AZURE_SEARCH_ENDPOINT`, "
        "`AZURE_SEARCH_API_KEY`, and `AZURE_SEARCH_INDEX_NAME` before starting the app."
    )
    st.stop()

if INDEX_STATUS_KEY not in st.session_state:
    get_cached_search_client()
    st.session_state[INDEX_STATUS_KEY] = "Connected to Azure AI Search."

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
            search_client = get_cached_search_client()
            chain = get_chain()
            chat_history = get_chat_history(st.session_state.messages[:-1])
            # Sources are fetched explicitly for display so the UI can expose
            # provenance independent of the final answer wording.
            docs = multi_stage_azure_retrieval(
                user_query,
                search_client=search_client,
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
