#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), ("human", "{text}")
])

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

messages = chat_prompt.format_messages(input_language="英文", output_language="中文", text="I love Large Language Model.")
print(messages)
res = model.invoke(messages)
print(res)
