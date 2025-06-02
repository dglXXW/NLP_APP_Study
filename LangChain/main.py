from GetApiKey import get_api_key
from Custom_Retriver import custom_retiver
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from base_func import pretty_print_docs
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os

# 模型本地环境配置
MODEL_PATH = os.environ.get("MODEL_PATH", "THUDM/chatglm3-6b")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# 本地ChatGLM3服务地址
endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

# PDF文件路径
PDF_path = "/root/autodl-tmp/data/Rag_File/实用程序育儿法.pdf"
# 向量数据库存储目录
persist_directory = "./chroma_langchain_db"
# 向量数据库检索Top_k
top_k_indexing = 3

if __name__ == "__main__":
    # 加载API KEY
    get_api_key()

    # 加载嵌入模型
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    """ 
        当指定 persist_directory 和 collection_name 时，Chroma 会尝试从磁盘加载同名集合。
        custom_retiver中封装了批量向数据库中增加数据的接口
    """
    # 构建向量数据库
    vector_store = custom_retiver(
        persist_directory=persist_directory,
        embeddings=embeddings,
        collection_name="RAG_Baby",
    ).get_vectorDB()

    print(f"向量数据库文档条目数：{vector_store._collection.count()}")

    question = "二月龄宝宝的肠胀气有哪些表现"

    """通过API调用本地服务"""
    llm = ChatGLM3(
        endpoint_url=endpoint_url, max_tokens=4096, top_p=0.7, temperature=0.8
    )

    """
        通过LLM实现文本压缩技术，缩减知识库检索后的文本上下文长度
    """
    # compressor = LLMChainExtractor.from_llm(llm)

    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=vector_store.as_retriever(
    #         search_type="mmr",
    #         search_kwargs={
    #             "k" : 5,
    #             "fetch_k" : 10
    #         }
    #     )
    # )

    # question = "宝宝肠胀气有什么表现，如何解决？"

    # docs = compression_retriever.invoke(question)

    # pretty_print_docs(docs)

    """
        问答
    """
    """构造提示词模板"""
    # template = """使用以下上下文片段来回答最后的问题。如果你不知道答案，只需说不知道，不要试图编造答案。尽量简明扼要地回答。在回答的最后一定要说"感谢您的提问！"
    # {context}
    # 问题：{question}
    # 有用的回答："""
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vector_store.as_retriever(
    #         search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
    #     ),
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    # )

    # question = "宝宝二月份肠胀气的表现是什么，有哪些解决措施？"

    # result = qa_chain.invoke({"query": question})
    # print(result["result"])

    """
        对话检索链
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history", # 与 prompt 的输入变量保持一致。
        return_messages=True # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )    

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手"),
        MessagesPlaceholder("chat_history"),  # 与 memory.memory_key 一致
        ("human", "{input}")
    ])
    

    conversation_qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
        ),
        memory=memory
    )


    question = "宝宝二月份肠胀气的表现是什么?"

    result = conversation_qa.invoke(question)

    print(result["output"])

    question = "有哪些解决措施？"

    result = conversation_qa.invoke(question)

    print(result["output"])
