from GetApiKey import get_api_key
from Custom_Splitter import PDFLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma

# PDF文件路径
PDF_path = "/root/autodl-tmp/data/Rag_File/实用程序育儿法.pdf"
# 向量数据库存储目录
persist_directory = "./chroma_langchain_db"
# 向量数据库检索Top_k
top_k_indexing = 3
# 向量数据库indexing批次
batch_size_indexing = 64

if __name__ == "__main__":
    # 加载API KEY
    print(get_api_key())

    # 读取PDF并分割
    docs_list = PDFLoader(PDF_path)

    # 构建向量数据库
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    # 当指定 persist_directory 和 collection_name 时，Chroma 会尝试从磁盘加载同名集合。
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="RAG_Baby",
        persist_directory=persist_directory,
    )

    # 每次运行程序时，清除selection中的数据
    vector_store.reset_collection()
    print(vector_store._collection.count())
    # 数据库存储数据Id
    docs_ids = []
    for i in range(0, len(docs_list), batch_size_indexing):
        batch_docs = docs_list[i : i + batch_size_indexing]

        docs_ids.extend(vector_store.add_documents(documents=batch_docs))

    print(len(docs_ids))

    print(vector_store._collection.count())

    question = "二月龄宝宝的肠胀气有哪些表现"

    docs = vector_store.max_marginal_relevance_search(question, k=top_k_indexing, fetch_k=5)
    # docs = vector_store.similarity_search(question, k=top_k_indexing)

    print(docs)
