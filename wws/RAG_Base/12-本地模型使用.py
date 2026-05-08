#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_huggingface import HuggingFaceEmbeddings

# 创建嵌入模型 —— 使用 HuggingFace Hub 自动下载
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

res = embeddings.embed_documents(["你好", "你"])
print(res)
print(embeddings.embed_query("你好"))
