#!/usr/bin/env python
# -*- coding: UTF-8 -*-



from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://api.deepseek.com", model="deepseek-chat")

res = llm.invoke('什么是大模型')
print(res.content)













