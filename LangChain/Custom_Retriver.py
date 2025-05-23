"""Embedding"""

from langchain_community.embeddings import ZhipuAIEmbeddings
from GetApiKey import get_api_key
from langchain_core.vectorstores import InMemoryVectorStore

if __name__ == "__main__":
    get_api_key()
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    # text = (
    #     "LangChain is the framework for building context-aware reasoning applications"
    # )

    # vectorstore = InMemoryVectorStore.from_texts(
    #     [text],
    #     embedding=embeddings,
    # )

    # # Use the vectorstore as a retriever
    # retriever = vectorstore.as_retriever()

    # # Retrieve the most similar text
    # retrieved_documents = retriever.invoke("What is LangChain?")

    # # show the retrieved document's content
    # print(retrieved_documents[0].page_content)

    # single_vector = embeddings.embed_query(text)
    # print(str(single_vector)[:100])  # Show the first 100 characters of the vector
