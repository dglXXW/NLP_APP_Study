from GetApiKey import get_api_key
from Custom_Splitter import PDFLoader

PDF_path = "/root/autodl-tmp/data/Rag_File/实用程序育儿法.pdf"

if __name__ == "__main__":
    # 加载API KEY
    print(get_api_key())

    # 读取PDF并分割
    docs_list = PDFLoader(PDF_path)
