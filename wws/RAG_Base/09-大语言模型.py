#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# LLM 纯文本补全模型（使用 DeepSeek 的 chat 模型代替千问的纯文本模型）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

text = "我的真的好想（帮我补全这个文本）"
res = llm.invoke(text)
print(res.content)
