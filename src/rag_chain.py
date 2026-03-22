from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

from src.retrieval_pipeline import multi_stage_retrieval

# prompt design for LLM to answer
# citation and source
# hallucination
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a machine learning expert and tutor.

        Your task is to answer the question using ONLY the provided context.
        Use the conversation history only to resolve references such as "it",
        "that method", or "the previous chapter".

        Guidelines:
        - Use ONLY the information from the context
        - Do NOT use your own knowledge
        - If the answer is not in the context, say: "I don't know based on the provided document"
        - Explain concepts simply
        - Provide examples only if they are supported by the context
        - ALWAYS cite your sources using this format:
          (Source: Chapter X | page Y)
        - If multiple sources are used, cite all of them
        - Be clear and concise

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
])


llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(
        f"Source:[{doc.metadata.get('metadata_label', 'Unknown source')}]\n Content:{doc.page_content}" for doc in docs
    )

def format_chat_history(chat_history):
    if not chat_history:
        return "No prior conversation."

    formatted_messages = []
    for message in chat_history:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def build_rag_chain(vector_store):
    retriever = RunnableLambda(
        lambda data: multi_stage_retrieval(
            data["question"],
            vector_store=vector_store,
            chat_history=data.get("chat_history", []),
        )
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnableLambda(lambda data: data["question"]),
            "chat_history": RunnableLambda(
                lambda data: format_chat_history(data.get("chat_history", []))
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
