#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


# 获取文档
loader = WebBaseLoader(
    'https://www.gov.cn/zhengce/content/202510/content_7043916.htm',
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id='UCAP-CONTENT')),
    requests_kwargs={"headers": {"User-Agent": "LangChainDemo/1.0"}},
)

# 文档切片
text_spl = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_spl.split_documents(loader.load())
docs = docs + docs + docs + docs  # 扩充文档数量用于测试批量处理
print(len(docs))

# 文档转换成向量 —— 本地 BGE 中文嵌入模型，无需 API Key
embs = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

vect = None
batch_size = 10

for i in range(0, len(docs), batch_size):
    batch_docs = docs[i:i + batch_size]
    print(f'第{i // batch_size + 1}批次 文档数量: {len(batch_docs)}')
    if i == 0:
        vect = FAISS.from_documents(batch_docs, embs)
    else:
        new_vect = FAISS.from_documents(batch_docs, embs)
        vect.merge_from(new_vect)

vect.save_local('faiss_store')
