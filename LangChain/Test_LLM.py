from openai import OpenAI
from GetApiKey import get_api_key, get_base_url
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.llms.chatglm3 import ChatGLM3
import os

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

def getRes_OpenAI_API(user_input, use_stream=False):
    client = OpenAI(api_key=get_api_key(local_llm=True), base_url=get_base_url())

    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=[
            {
                "role": "system",
                "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's "
                        "instructions carefully. Respond using markdown.",
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        stream=use_stream,
        top_p=0.8,
        temperature=0.9,
    )

    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


def getRes_ZhipuAI(user_input):
    """
    无法调用本地的ChatGLM3-6B的API服务
    """
    chat = ChatZhipuAI(
        model="chatglm3-6b",
        temperature=0.9,
        api_key=get_api_key(False),
        api_base=get_base_url(),
    )

    messages = [
        SystemMessage(content="Your role is a poet."),
        HumanMessage(content=user_input),
    ]

    response = chat.invoke(messages)
    print(response.content)  # Displays the AI-generated poem

def getRes_ChatOpenAI(user_input):
    chat = ChatOpenAI(
        model="chatglm3-6b",
        temperature=0.9,
        openai_api_key=get_api_key(True),
        openai_api_base=get_base_url()
    )

    messages = [
        SystemMessage(content="Your role is a poet."),
        HumanMessage(content=user_input),
    ]

    response = chat.invoke(messages)
    print(response)

def getRes_ChatGLM3(user_input):

    messages = [
        SystemMessage(content="你是一个智能助手"),
        HumanMessage(content=user_input),
    ]
        
    """通过API调用本地服务"""
    endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
    llm = ChatGLM3(
        endpoint_url=endpoint_url,
        max_tokens=4096,
        top_p=0.9
    )
    """直接加载本地模型 
    TODO 即使加载本地模型，也需要启动服务，为什么？"""
    # llm = ChatGLM3(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH ,device="cuda")
    response = llm.invoke(messages)
    print(response)

if __name__ == "__main__":
    getRes_ChatGLM3("描述无锡这座城市，不多于30字")
