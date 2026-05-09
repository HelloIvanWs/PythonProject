#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
基于LangChain Runnable的旅游问答系统

该系统结合了语言模型、向量数据库和搜索工具，能够处理用户关于旅游景点的提问，
包括天气查询、景点介绍和行程规划等功能。使用Runnable组件构建可组合的处理管道。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# faiss  InMemoryVectorStore 基于内存的向量数据库
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain.schema.runnable import RunnableMap, RunnableBranch, RunnableLambda
# 导入Langchain的runnable组件
from langchain_core.runnables import RunnableMap, RunnableBranch, RunnableLambda
# 搜索工具
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class TravelQASystem:
    """
    旅游智能问答系统
    
    该系统整合了大型语言模型、向量检索和网络搜索功能，专门用于处理旅游相关咨询。
    支持景点信息查询、天气查询和行程规划建议等场景。
    
    Attributes:
        llm: ChatOpenAI语言模型实例，用于生成回答和解析用户意图
        search: TavilySearch搜索引擎实例，用于获取实时天气等信息
        embeddings: HuggingFace嵌入模型实例，用于文本向量化
        attraction_data: 景点信息数据列表，包含景点名称、特点和开放时间等信息
        vector_store: InMemoryVectorStore向量存储实例，用于相似度检索
        travel_qa_pipeline: LangChain Runnable处理管道，定义完整的问答流程
    """
    
    def __init__(self, openai_api_key, serpapi_api_key, embed_path):
        """
        初始化旅游问答系统核心组件
        
        配置系统所需的语言模型、搜索工具和嵌入模型，并构建景点知识库的向量索引。
        
        Args:
            openai_api_key (str): OpenAI API密钥，用于访问DashScope兼容接口
            serpapi_api_key (str): Tavily搜索API密钥，用于执行网络搜索
            embed_path (str): HuggingFace嵌入模型的本地路径，用于文本向量化
            
        Returns:
            None: 此方法不返回任何值，仅初始化实例属性
            
        Raises:
            Exception: 当API密钥无效或模型路径不存在时可能抛出异常
        """

        # 初始化语言模型
        self.llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )

        # 初始化搜索工具
        self.search = TavilySearch(tavily_api_key=serpapi_api_key)

        # 初始化嵌入模型
        # self.embeddings = HuggingFaceEmbeddings(model_name=embed_path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True},
        )

        # 构建景点知识库
        self.attraction_data = [
            "故宫：北京地标，明清皇宫，开放时间8:30-17:00",
            "颐和园：皇家园林，昆明湖、长廊等景点",
            "八达岭长城：距离市区70公里，建议游览3-4小时"
        ]

        # 使用内存型向量存储类
        self.vector_store = InMemoryVectorStore.from_texts(
            self.attraction_data, self.embeddings, k=1
        )

    def setup_runnable_pipeline(self):
        """
        定义Runnable流程管道
        
        构建一个多阶段的LangChain Runnable处理管道，包含以下阶段：
        1. 问题解析：从用户输入中提取地点和查询类型（JSON格式）
        2. 数据获取：根据查询类型并行获取天气信息和景点知识
        3. 分支处理：使用RunnableBranch根据查询类型选择不同的处理策略
        4. 回答生成：结合所有信息生成专业的旅游建议
        
        该管道采用声明式编程风格，支持链式调用和自动流转。
        
        Pipeline Components:
            parse_module: 问题解析模块，将自然语言转换为结构化JSON
            weather_query: 天气查询Lambda函数，调用Tavily搜索获取实时天气
            attraction_retrieve: 景点检索Lambda函数，从向量库中检索相关景点信息
            data_acquisition: 数据获取模块，并行执行多个数据源查询（RunnableMap）
            generate_prompt: 回答生成的提示模板，整合所有上下文信息
            generate_module: 最终的回答生成模块
            travel_qa_pipeline: 完整的问答处理管道（实例属性）
            
        Returns:
            None: 此方法不返回任何值，设置self.travel_qa_pipeline实例属性
            
        Note:
            - 必须在调用process_user_question之前先调用此方法
            - RunnableBranch根据查询类型是否包含'天气'关键字进行分支判断
            - data_acquisition使用RunnableMap实现并行数据获取，提高响应效率
        """
        # 3.1 问题解析模块：识别地点与查询类型
        parse_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是旅游助手，需从用户问题中提取地点和查询类型（天气/景点介绍/行程规划）"),
            ("user", """问题：{user_question}请以JSON格式返回：{{"location": "地点", "type": "查询类型"}}""")
        ])
        parse_module = parse_prompt | self.llm | JsonOutputParser()  # Output JSON string

        # 搜索天气
        weather_query = RunnableLambda(
            lambda x: self.search.invoke(f"{x['location']}的天气怎么样")
        )
        
        # 检索数据
        attraction_retrieve = (lambda x: x['location']) | self.vector_store.as_retriever() | (
            lambda x: x[0].page_content)

        # RunnableMap  并行执行
        data_acquisition = RunnableMap({
            "weather": weather_query,
            'attraction': attraction_retrieve,
            'location': (lambda x: x['location'])
        })

        generate_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是专业旅游顾问，需结合景点信息和天气生成建议"),
            ("user", """地点：{location}
                        景点信息：{attraction}
                        天气情况：{weather}
                        请生成1条行程建议，包含注意事项（如天气相关准备）""")
        ])

        generate_module = generate_prompt | self.llm | (lambda x: x.content.strip())

        # RunnableBranch   根据查询的查询类型进行分支处理
        self.travel_qa_pipeline = (parse_module |
                                   (lambda x: {"location": x['location'], "type": x['type']}) |
                                   RunnableBranch(
                                       (lambda x: '天气' in x['type'], data_acquisition),
                                       lambda x: {"location": x["location"],
                                                  "attraction": attraction_retrieve.invoke(x)}
                                   )
                                   | generate_module
                                   )


    def process_user_question(self, user_question):
        """
        处理用户提问并返回回答
        
        接收用户的自然语言问题，通过预定义的Runnable管道进行处理，
        返回AI生成的旅游建议或信息。
        
        Args:
            user_question (str): 用户的旅游相关问题，例如"今天故宫的天气怎么样?"
            
        Returns:
            str: AI生成的回答内容，包含旅游建议、天气信息或景点介绍等
            
        Example:
            >>> travel_qa = TravelQASystem(api_key, search_key, embed_path)
            >>> travel_qa.setup_runnable_pipeline()
            >>> response = travel_qa.process_user_question("故宫有什么好玩的?")
            >>> print(response)
        """
        input_data = {"user_question": user_question}
        # try:
        response = self.travel_qa_pipeline.invoke(input_data)
        return response


# 示例用法
if __name__ == "__main__":
    # 替换为实际API密钥
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    # https://www.tavily.com/
    SERPAPI_API_KEY = os.getenv("TAVILY_API_KEY")
    embed_path = r"D:\LLM\Local_model\BAAI\bge-large-zh-v1___5"

    # 初始化系统
    travel_qa = TravelQASystem(OPENAI_API_KEY, SERPAPI_API_KEY, embed_path)
    travel_qa.setup_runnable_pipeline()

    # 测试1：查询天气与景点建议
    question1 = "今天故宫的天气怎么样?"
    answer1 = travel_qa.process_user_question(question1)
    print(f"User Question: {question1}\nAI Answer: {answer1}\n")
