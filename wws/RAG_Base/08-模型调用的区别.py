#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# 通过 init_chat_model 调用 DeepSeek
llm = init_chat_model(
    "deepseek-chat",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

res = llm.invoke('什么是大模型')
print(res)
print(res.content)

from langchain_openai import ChatOpenAI

# 通过 langchain_openai 调用 DeepSeek（OpenAI 兼容）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

response = llm.invoke("什么是大模型？")
print(response)
print("=" * 50)
print(response.content)

# 用 DeepSeek 厂商自己的封装调用
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

response = llm.invoke("什么是大模型？")
print(response)
print("=" * 50)
print(response.content)
