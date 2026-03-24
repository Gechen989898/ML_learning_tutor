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
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_rerank_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retrieve_candidates(query, k, vector_store):
    return vector_store.similarity_search(query, k=k)


def parse_indices(text):
    cleaned_text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        numbers = re.findall(r"\d+", cleaned_text)
        return [int(n) for n in numbers]


def format_chat_history(chat_history):
    if not chat_history:
        return "No prior conversation."

    formatted_messages = []
    for message in chat_history:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def rewrite_query(question, chat_history=None):
    if not chat_history:
        return question

    history_text = format_chat_history(chat_history)
    response = get_rewrite_llm().invoke(
        rewrite_prompt.format(chat_history=history_text, question=question)
    )
    rewritten_question = response.content.strip()
    return rewritten_question or question


def rerank_docs(documents, query, top_k):
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
    standalone_query = rewrite_query(query, chat_history=chat_history)
    candidates = retrieve_candidates(standalone_query, k=candidate_k, vector_store=vector_store)
    return rerank_docs(candidates, standalone_query, top_k=top_k)
