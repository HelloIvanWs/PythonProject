#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from langchain_huggingface import HuggingFaceEmbeddings

# 初始化本地 BGE 嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

# 文档列表嵌入
doc_res = embeddings.embed_documents(["你好", "世界"])
print(doc_res)
print(len(doc_res))

# 单条查询嵌入
res = embeddings.embed_query("你好")
print(res)
