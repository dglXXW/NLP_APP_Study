"""获取API_KEY"""

import os
from dotenv import load_dotenv, find_dotenv


def get_api_key():
    _ = load_dotenv(find_dotenv())
    return os.environ["ZHIPUAI_API_KEY"]


if __name__ == "__main__":
    print(get_api_key())
