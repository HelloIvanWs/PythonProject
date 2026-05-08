
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# 获取文档
loder = WebBaseLoader('https://www.jianshu.com/p/ac66a23cc683',bs_kwargs=dict(parse_only=bs4.SoupStrainer(id='__next')))

print('获取文档=============================================================================================')

# 文档切片
test_spl =  RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = test_spl.split_documents(loder.load())

print('文档切片=============================================================================================')
# 文档转为向量
# 初始化本地 BGE 嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)
embedding_res = embeddings.embed_documents([doc.page_content for doc in docs])
print(len(embedding_res))
print('文档转为向量=============================================================================================')
vect = FAISS.from_documents(docs, embeddings)
vect.save_local("faiss_store")