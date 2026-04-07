"""CLI entrypoint for querying the textbook-backed RAG system."""

from dotenv import load_dotenv

from learning_tutor.azure_search import get_search_client
from learning_tutor.rag_chain import build_azure_rag_chain


load_dotenv()


if __name__ == "__main__":
    search_client = get_search_client()
    chain = build_azure_rag_chain(search_client)
    print("Connected to Azure AI Search.")
    print("Ask a question about the book.")
    user_query = input("Question: ")
    response = chain.invoke({"question": user_query, "chat_history": []})
    print(response)
