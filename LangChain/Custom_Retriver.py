"""Embedding"""

from langchain_community.embeddings import ZhipuAIEmbeddings
from GetApiKey import get_api_key
from langchain_core.vectorstores import InMemoryVectorStore
import numpy as np
from langchain_chroma import Chroma
from Custom_Splitter import PDFLoader


class custom_retiver:

    def __init__(self, persist_directory, embeddings, collection_name):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )


    def similarity_search(self, query, top_k, filters, **kwargs):
        """向量数据库有两种基本的索引模式 相似性搜索和最大边际相关性搜索"""
        # docs = vector_store.max_marginal_relevance_search(question, k=top_k_indexing, fetch_k=5)
        # docs = vector_store.similarity_search(question, k=top_k_indexing)
        return self.vector_store.similarity_search(query, top_k, filters)

    # batch_size_indexing 向量数据库indexing批次
    def add_documents(self, pdf_path, batch_size_indexing):
        # 读取PDF并分割
        docs_list = PDFLoader(pdf_path)
        # 数据库存储数据Id
        docs_ids = []
        # 根据embedding model每batch最大处理的索引数量设置batch_size_indexing作为迭代步长
        # 每次批量向数据库中增加batch_size_indexing条数据
        for i in range(0, len(docs_list), batch_size_indexing):
            batch_docs = docs_list[i : i + batch_size_indexing]

            docs_ids.extend(vector_store.add_documents(documents=batch_docs))

        return doc_ids
    
    def get_vectorDB(self):
        return self.vector_store

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
