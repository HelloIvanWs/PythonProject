import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

ENDPOINT_URL = "Qwen/Qwen3-8B"
HF_TOKEN = os.getenv('HF_TOKEN')

llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    huggingfacehub_api_token=HF_TOKEN,
)
chat_model = ChatHuggingFace(llm=llm)
resp = chat_model.invoke("解释 prompt 是什么？")
print(resp)
