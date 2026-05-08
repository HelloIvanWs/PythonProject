import os

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 初始化本地 BGE 嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

# 加载本地FAISS索引
save_path = 'faiss_store'

vect_store = FAISS.load_local(
    folder_path=save_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True # 允许加载pickle文件
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_template("""
仅根据提供的上下文回答以下问题：
<context>
{context}
</context>
问题：{input}
""")

llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"), model="deepseek-chat")

# 把拼接号的提示词给大模型
doc_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索器
retr = vect_store.as_retriever()

res_chain = create_retrieval_chain(retr, doc_chain)

res = res_chain.invoke({"input":"jstat命令的作用是什么"})
print(res)