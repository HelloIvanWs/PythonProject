import os
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 创建嵌入模型 —— 必须与构建索引时使用的模型一致
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

# 加载本地 FAISS 索引
vector_store = FAISS.load_local(
    folder_path="../faiss_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="deepseek-chat",
)

doc_chain = create_stuff_documents_chain(llm, prompt)
retr = vector_store.as_retriever()
res_chain = create_retrieval_chain(retr, doc_chain)

res = res_chain.invoke({"input": "密云水库水源保护条例什么时候执行"})
print(res)
