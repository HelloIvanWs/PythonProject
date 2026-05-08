#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

human_text = "你好啊"
human_text2 = "给我讲个笑话"
system_text = "你是一个强大的助手，你的名字叫0713"

# 聊天模型（DeepSeek）
chat_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

# messages = [HumanMessage(content=human_text)]
# 聊天模型支持多个消息作为输入
messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]

res = chat_model.invoke(messages)
print(res)

print('-----------------')
# 聊天模型支持多个消息作为输入
messages2 = [SystemMessage(content=system_text), HumanMessage(content=human_text), HumanMessage(content=human_text2)]
res2 = chat_model.invoke(messages2)
print(res2)


