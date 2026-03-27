"""Retrieval utilities for query rewriting, candidate search, and reranking."""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Given the conversation history and the latest user question, rewrite the latest
    question so it can be understood on its own.

    Keep the meaning unchanged.
    If the latest question is already standalone, return it unchanged.

    Conversation history:
    {chat_history}

    Latest question:
    {question}

    Standalone question:
    """
)

rerank_prompt = ChatPromptTemplate.from_template(
    """
    You are a ranking assistant.

    Given a question and a list of documents, rank the documents by relevance.
    Return ONLY a valid JSON list of indices.
    Do NOT use markdown.

    Example:
    [0, 3, 5, 2, 1]

    Question:
    {question}

    Documents:
    {documents}

    Return the indices of the top {top_k} most relevant documents in order.
    """
)


def get_rewrite_llm():
    """Create the model used to rewrite follow-up questions.

    Query rewriting improves retrieval when the user asks conversational
    follow-ups that omit important nouns or referents from earlier turns.

    Args:
        None

    Returns:
        ChatOpenAI: Deterministic chat model for standalone query generation.
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_rerank_llm():
    """Create the model used to rerank retrieved documents.

    The reranker is separated from the initial vector search so the system can
    use embeddings for broad recall and an LLM for finer-grained relevance.

    Args:
        None

    Returns:
        ChatOpenAI: Deterministic chat model for document reranking.
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retrieve_candidates(query, k, vector_store):
    """Retrieve an initial set of candidate documents from FAISS.

    This stage optimizes for recall. It intentionally fetches more documents
    than will be shown to the answer model so reranking has enough evidence to
    recover relevant passages that may not be first by cosine similarity alone.

    Args:
        query: Standalone user query used for retrieval.
        k: Number of nearest-neighbor candidates to return.
        vector_store: FAISS vector store containing embedded chunks.

    Returns:
        list: Candidate documents returned by similarity search.

    Notes:
        Larger ``k`` can improve recall but increases reranking latency and
        prompt size.
    """
    return vector_store.similarity_search(query, k=k)


def parse_indices(text):
    """Parse reranker output into a list of document indices.

    The reranker is instructed to emit JSON, but the parser also tolerates minor
    formatting drift so the retrieval pipeline remains robust to imperfect model
    outputs.

    Args:
        text: Raw model output from the reranker.

    Returns:
        list[int]: Ranked document indices extracted from the response.
    """
    cleaned_text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        numbers = re.findall(r"\d+", cleaned_text)
        return [int(n) for n in numbers]


def format_chat_history(chat_history):
    """Render chat history into a plain-text transcript.

    Args:
        chat_history: Prior user and assistant turns.

    Returns:
        str: Formatted transcript used by prompts.
    """
    if not chat_history:
        return "No prior conversation."

    formatted_messages = []
    for message in chat_history:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def rewrite_query(question, chat_history=None):
    """Rewrite a follow-up question into a standalone retrieval query.

    Args:
        question: Latest user question.
        chat_history: Optional prior conversation turns.

    Returns:
        str: Standalone query used for retrieval.
    """
    if not chat_history:
        return question

    history_text = format_chat_history(chat_history)
    response = get_rewrite_llm().invoke(
        rewrite_prompt.format(chat_history=history_text, question=question)
    )
    rewritten_question = response.content.strip()
    return rewritten_question or question


def rerank_docs(documents, query, top_k):
    """Rerank candidate documents and keep the strongest evidence.

    Vector retrieval is efficient but approximate. Reranking is applied to
    improve precision before answer generation, which helps reduce hallucination
    risk by passing a smaller, more relevant context window to the final model.

    Args:
        documents: Candidate documents returned by vector search.
        query: Standalone retrieval query.
        top_k: Number of reranked documents to keep.

    Returns:
        list: Top reranked documents, or a fallback slice of the input list.

    Notes:
        Lower ``top_k`` reduces noise and latency, while higher ``top_k`` may
        preserve more recall at the cost of answer precision.
    """
    format_text = "\n\n".join(
        f"[{i}]\n{doc.metadata.get('metadata_label', 'Unknown source')}\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )
    response = get_rerank_llm().invoke(
        rerank_prompt.format(
            question=query,
            documents=format_text,
            top_k=top_k,
        )
    )
    indices = parse_indices(response.content)
    valid_indices = []
    for index in indices:
        if 0 <= index < len(documents) and index not in valid_indices:
            valid_indices.append(index)
    reranked_docs = [documents[index] for index in valid_indices[:top_k]]
    return reranked_docs if reranked_docs else documents[:top_k]


def multi_stage_retrieval(
    query,
    vector_store,
    chat_history=None,
    candidate_k=20,
    top_k=5,
):
    """Run the full retrieval pipeline for one user query.

    The pipeline first rewrites conversational questions, then retrieves a
    recall-oriented candidate pool from FAISS, and finally reranks those
    candidates for precision. This two-stage retrieval design is a practical
    latency versus accuracy trade-off for small RAG systems.

    Args:
        query: User question to answer.
        vector_store: FAISS vector store containing embedded chunks.
        chat_history: Optional prior conversation turns.
        candidate_k: Number of documents to retrieve before reranking.
        top_k: Number of reranked documents to return.

    Returns:
        list: Documents selected for the answer-generation stage.

    Notes:
        ``candidate_k`` mainly controls recall, while ``top_k`` mainly controls
        precision, prompt size, and downstream answer latency.
    """
    standalone_query = rewrite_query(query, chat_history=chat_history)
    candidates = retrieve_candidates(
        standalone_query,
        k=candidate_k,
        vector_store=vector_store,
    )
    return rerank_docs(candidates, standalone_query, top_k=top_k)
