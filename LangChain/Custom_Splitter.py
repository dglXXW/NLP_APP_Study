"""文档分割器"""

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)


# text_r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=20, chunk_overlap=3, separators=["\n\n", "(?<=。)", "(?<=！)", "(?<=？)", "(?<=，)", "(?<=；)", " ", ""],
#     is_separator_regex=True
# )

# text1 = """在编写文档时，作者将\n使用文档结构对\n内容进行分组！ \
#     这可以\n向读者传达哪些想法是相\n关的； 例如，密切相\n关的想法\
#     是在句子中。 类似\n的想法在段落中。 段落构成文\n档。 \n\n\
#     段落通常用一\n个或两个回车符分隔。\n \
#     回车符是您在\n该字符串中看到的嵌\n入的“反斜杠 n”。 \
#     句子末尾有一\n个句号，但也有一个空格\n。\
#     并且单词之间\n用空格分隔"""

# print(text_r_splitter.split_text(text1))


def PDFLoader(path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """加载pdf文档"""
    # 创建一个PyPDFLoader实例并加载pdf文档
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(
        f"loader.load()返回类型为{type(pages)} 每一页的类型为{type(pages[0])} 长度为{len(pages)}"
    )

    # for idx, page in enumerate(pages):
    #     if page.page_content:
    #         print(
    #             f"第{idx}页文档: 该页文档长度：{len(page.page_content)} 元数据为{page.metadata}"
    #         )

    """ 分割已加载的Document对象 """

    # 初始化文本分割器
    r_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "(?<=。\n)",
            "(?<=。)",
            "(?<=！)",
            "(?<=？)",
            "(?<=，)",
            "(?<=；)",
            " ",
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True,
    )
    """采用split_text分割"""
    # pages_segments = ''
    # for idx in range(5):
    #     pages_segments +=  pages[idx].page_content

    # docs_list = r_splitter.split_text(pages_segments)
    # print(f"分割后文本块数量为{len(docs_list)}")

    # for idx in range(len(docs_list)):
    #     print(type(docs_list[idx]))
    #     print(docs_list[idx])

    """采用split_documents分割"""
    docs_splits = r_splitter.split_documents(pages)
    print(f"分割后文本块数量为{len(docs_splits)}")
    return docs_splits
    # for idx in range(10):
    #     print(type(docs_splits[idx]))
    #     print(docs_splits[idx].page_content)
