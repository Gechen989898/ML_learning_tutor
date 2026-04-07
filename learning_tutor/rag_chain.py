"""Chain assembly for grounded answer generation over retrieved chunks."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from learning_tutor.retrieval_pipeline import (
    multi_stage_azure_retrieval,
    multi_stage_retrieval,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful tutor answering questions about a textbook.
            Use only the provided context to answer the question.
            Do not use outside knowledge.

            Citation rules:
            - Cite only the sources that directly support your answer.
            - Do not cite every source automatically.
            - Use the exact source label shown in the context.
            - Put citations at the end of the sentence or paragraph they support.
            - If multiple sources support the same point, cite all relevant sources.

            If the context does not contain enough information to answer, say:
            "I don't know based on the provided document."

            Keep the explanation clear, simple, and concise.
            """
        ),
        (
            "human",
            """
            Conversation history:
            {chat_history}

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        ),
    ]
)


def get_llm():
    """Create the answer-generation model for the RAG chain.

    Args:
        None

    Returns:
        ChatOpenAI: Chat model used to generate grounded answers.
    """
    return ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs):
    """Format retrieved documents into prompt-ready context.

    Source labels are preserved in the context block so the generation model can
    cite evidence explicitly in the final answer.

    Args:
        docs: Documents selected by the retrieval pipeline.

    Returns:
        str: Concatenated context string with source labels.
    """
    return "\n\n".join(
        f"Source: {doc.metadata.get('metadata_label', 'Unknown source')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def format_chat_history(chat_history):
    """Render prior turns into a plain-text transcript.

    Args:
        chat_history: Prior user and assistant turns.

    Returns:
        str: Formatted chat transcript for prompt injection.
    """
    if not chat_history:
        return "No prior conversation."

    formatted_messages = []
    for message in chat_history:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def build_rag_chain(vector_store):
    """Build the end-to-end retrieval-augmented generation chain.

    The prompt is intentionally restrictive: it tells the model to answer only
    from retrieved context, cite its evidence, and admit when the context is
    insufficient. This is the main hallucination mitigation strategy in the
    generation layer.

    Args:
        vector_store: FAISS vector store backing retrieval.

    Returns:
        Runnable: LangChain runnable that retrieves context and generates an answer.

    Notes:
        Strict grounding improves factual reliability, but it can make the
        system more conservative when retrieval misses relevant context.
    """
    # Retrieval is invoked inside the chain so the latest query and chat
    # history stay synchronized with the final generation request.
    retriever = RunnableLambda(
        lambda data: multi_stage_retrieval(
            data["question"],
            vector_store=vector_store,
            chat_history=data.get("chat_history", []),
        )
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnableLambda(lambda data: data["question"]),
            "chat_history": RunnableLambda(
                lambda data: format_chat_history(data.get("chat_history", []))
            ),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_azure_rag_chain(
    search_client,
    candidate_k=20,
    top_k=5,
    use_semantic=True,
    semantic_config_name=None,
):
    """Build a RAG chain backed by Azure AI Search hybrid/semantic retrieval.

    Args:
        search_client: Azure AI Search document client.
        candidate_k: Number of vector nearest neighbors to retrieve.
        top_k: Number of search results to include in the answer context.
        use_semantic: Whether to enable Azure semantic ranking.
        semantic_config_name: Optional semantic configuration name.

    Returns:
        Runnable: LangChain runnable that retrieves context and generates an answer.
    """
    retriever = RunnableLambda(
        lambda data: multi_stage_azure_retrieval(
            data["question"],
            search_client=search_client,
            chat_history=data.get("chat_history", []),
            candidate_k=candidate_k,
            top_k=top_k,
            use_semantic=use_semantic,
            semantic_config_name=semantic_config_name,
        )
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnableLambda(lambda data: data["question"]),
            "chat_history": RunnableLambda(
                lambda data: format_chat_history(data.get("chat_history", []))
            ),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
