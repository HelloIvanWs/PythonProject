#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

prompt = PromptTemplate(
    template="您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
)

aa = "您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
print(aa.format(text="python"))
inputs = prompt.format(text="python")
print(inputs)

oup = model.invoke(inputs)
print(oup)
