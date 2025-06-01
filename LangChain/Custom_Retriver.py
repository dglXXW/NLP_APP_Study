"""Embedding"""

from langchain_community.embeddings import ZhipuAIEmbeddings
from GetApiKey import get_api_key
from langchain_core.vectorstores import InMemoryVectorStore
import numpy as np
from langchain_chroma import Chroma

persist_directory = "./chroma_langchain_db"


class custom_retiver:

    def __init__(self, persist_directory, embeddings, collection_name):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def similarity_search(self, query, top_k, filters, **kwargs):
        return self.vector_store.similarity_search(query, top_k, filters)


if __name__ == "__main__":
    get_api_key()
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    sentence1_chinese = "猫是猫科动物"
    sentence2_chinese = "狗是犬科动物"
    sentence3_chinese = "眼睛看不见"

    # embedding1_chinese = embeddings.embed_query(sentence1_chinese)
    # embedding2_chinese = embeddings.embed_query(sentence2_chinese)
    # embedding3_chinese = embeddings.embed_query(sentence3_chinese)

    # print(np.dot(embedding1_chinese, embedding2_chinese))

    # print(np.dot(embedding1_chinese, embedding3_chinese))

    # print(np.dot(embedding2_chinese, embedding3_chinese))

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings
    )

    print(vector_store._collection.count())
    doc_ids = vector_store.add_texts([sentence1_chinese, sentence2_chinese, sentence3_chinese])
    print(doc_ids)
    question = "狗是什么？"
    # docs = vector_store.similarity_search(question, k=1)

    list_scores = vector_store.similarity_search_with_score(question, k=3)
    
    # for tuples in list_scores:
    #     doc = tuples[0]
    #     score = tuples[1]
    #     doc.metadata["score"] = score
    print(list_scores)
    print(vector_store._collection.count())
