"""获取API_KEY"""

import os
from dotenv import load_dotenv, find_dotenv


def get_api_key(local_llm = False):
    if not local_llm:
        _ = load_dotenv(find_dotenv())
        return os.environ["ZHIPUAI_API_KEY"]
    return "EMPTY"

def get_base_url():
    return "http://127.0.0.1:8000/v1/"

if __name__ == "__main__":
    print(get_api_key())
