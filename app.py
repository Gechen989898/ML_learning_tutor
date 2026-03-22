from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import streamlit as st

from src.data_pipeline import load_data, split_chunk, split_clean_chunks
from src.embedding import embedding_vector, load_vector_store, save_vector_store
from src.rag_chain import build_rag_chain
from src.retrieval_pipeline import multi_stage_retrieval

DEFAULT_FILE_PATH = "data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf"
DEFAULT_INDEX_DIR = "storage/faiss_index"
DEFAULT_TITLE = "Learning Tutor"
DEFAULT_GREETING = (
    "Ask me anything about the book."
)
INDEX_STATUS_KEY = "index_status"


def build_vector_store(file_path):
    document = load_data(file_path)
    filtered_docs = split_chunk(document)
    chunks = split_clean_chunks(filtered_docs)
    return embedding_vector(chunks)


@st.cache_resource(show_spinner=False)
def get_vector_store(file_path, index_dir):
    index_path = Path(index_dir)
    vector_store = load_vector_store(index_dir)
    if vector_store is not None:
        return vector_store, f"Loaded existing FAISS index from `{index_path}`."

    vector_store = build_vector_store(file_path)
    save_vector_store(vector_store, index_dir)
    return vector_store, f"Built a new FAISS index and saved it to `{index_path}`."


@st.cache_resource(show_spinner=False)
def get_chain(file_path, index_dir):
    vector_store, _ = get_vector_store(file_path, index_dir)
    return build_rag_chain(vector_store)


def format_sources(docs):
    labels = []
    for doc in docs:
        label = doc.metadata.get("metadata_label", "Unknown source")
        if label not in labels:
            labels.append(label)
    return labels


def get_chat_history(messages):
    return [message for message in messages if message["role"] in {"user", "assistant"}]


st.set_page_config(page_title=DEFAULT_TITLE, layout="wide")
st.title(DEFAULT_TITLE)
st.caption("Chat with your book-backed RAG pipeline.")

file_path = DEFAULT_FILE_PATH
index_dir = DEFAULT_INDEX_DIR

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
