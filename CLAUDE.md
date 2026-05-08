# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a LangChain tutorial workspace — 14 progressively numbered Python scripts that teach LangChain concepts. It's a PyCharm project using Python 3.13 with a virtual environment at `.venv/`.

## Repository structure

```
wws/
├── .env                          # API keys and base URLs
├── 笔记.txt                        # Study notes (RAG concepts, LangChain architecture)
├── faiss_store/                  # Pre-built FAISS vector index
│   ├── index.faiss
│   └── index.pkl
├── RAG_Base/                     # Original 13-script tutorial series
│   ├── 01-Langchain调用模型.py
│   ├── 02-构建索引.py
│   ├── 03-构建检索.py
│   ├── 04-字符串提示词模版.py
│   ├── 05-聊天提示词模版.py
│   ├── 06-聊天提示词.py
│   ├── 07-少量样本.py
│   ├── 08-模型调用的区别.py
│   ├── 09-大语言模型.py
│   ├── 10-聊天模型.py
│   ├── 11-文本嵌入模型.py
│   ├── 12-本地模型使用.py
│   └── 13-调用hug模型.py
└── RAG_LangChain/                # New LangChain practice module
    └── Langchain_llm.py
```

## Commands

No build, lint, or test tooling is configured. Scripts must be run from the `wws/` directory so that `load_dotenv()` finds `.env` and relative paths (e.g., `faiss_store`) resolve correctly:

```bash
source .venv/Scripts/activate
cd wws
python RAG_Base/01-Langchain调用模型.py
python RAG_LangChain/Langchain_llm.py
```

There is no `requirements.txt`. Install packages as needed with pip.

## Environment configuration

API keys and base URLs are loaded via `python-dotenv` from `wws/.env`. Each script calls `load_dotenv()` (without a path argument), so scripts must be run with `wws/` as the working directory.

The `.env` file contains:
- `DASHSCOPE_API_KEY` — repurposed as the DeepSeek API key
- `DASHSCOPE_BASE_URL` — repurposed as the DeepSeek base URL (`https://api.deepseek.com`)
- `HF_TOKEN` — HuggingFace Hub token (used by `13-调用hug模型.py`)

Note: Despite the `DASHSCOPE_` prefix, these keys now point to DeepSeek, not DashScope. The project has migrated from Qwen/DashScope to DeepSeek for LLM calls.

All scripts (including `RAG_LangChain/Langchain_llm.py`) now use `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`.

## Model and provider architecture

**LLM / Chat models — DeepSeek (via OpenAI-compatible API):**
All chat/LLM scripts use `langchain_openai.ChatOpenAI` pointed at `https://api.deepseek.com` with `model="deepseek-chat"`. The three model instantiation patterns demonstrated in `08-模型调用的区别.py`:
| Method | Import |
|--------|--------|
| `init_chat_model` | `langchain.chat_models` |
| `ChatOpenAI` | `langchain_openai` |
| `ChatDeepSeek` | `langchain_deepseek` |

**Embedding models — HuggingFace BGE (local):**
All embedding scripts use `HuggingFaceEmbeddings` with `BAAI/bge-small-zh-v1.5` (or `bge-large-zh-v1.5` in `12-本地模型使用.py`). Models download automatically on first use. No API key required.

**HuggingFace hosted inference — `13-调用hug模型.py`:**
Uses `HuggingFaceEndpoint` + `ChatHuggingFace` to call `Qwen/Qwen3-8B` via HuggingFace Hub. Requires `HF_TOKEN` in `.env`.

## Script curriculum

### RAG_Base — core LangChain tutorial series

| File | Topic |
|------|-------|
| `01-Langchain调用模型.py` | Basic ChatOpenAI invocation with DeepSeek |
| `02-构建索引.py` | Web doc loading (WebBaseLoader), text splitting (RecursiveCharacterTextSplitter), local BGE embeddings, FAISS vector store with batched ingestion |
| `03-构建检索.py` | RAG retrieval chain: load FAISS index, create_stuff_documents_chain + create_retrieval_chain |
| `04-字符串提示词模版.py` | String PromptTemplate |
| `05-聊天提示词模版.py` | ChatPromptTemplate with system/human messages |
| `06-聊天提示词.py` | SystemMessagePromptTemplate + HumanMessagePromptTemplate composition |
| `07-少量样本.py` | FewShotPromptTemplate |
| `08-模型调用的区别.py` | Comparing `init_chat_model`, `ChatOpenAI`, and `ChatDeepSeek` (all pointing to DeepSeek) |
| `09-大语言模型.py` | LLM-style text completion via ChatOpenAI (DeepSeek) |
| `10-聊天模型.py` | Chat model with SystemMessage + HumanMessage objects |
| `11-文本嵌入模型.py` | Text embeddings via HuggingFace BGE |
| `12-本地模型使用.py` | Local embeddings with larger BGE model (`bge-large-zh-v1.5`) |
| `13-调用hug模型.py` | HuggingFace hosted inference via HuggingFaceEndpoint + ChatHuggingFace |

### RAG_LangChain — extended practice

| File | Topic |
|------|-------|
| `Langchain_llm.py` | Simple ChatOpenAI call using DeepSeek (DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL) |

## Key patterns

- Every script calls `load_dotenv()` at the top (no path argument — requires running from `wws/`)
- Models are instantiated per-script (no shared config module)
- `02-构建索引.py` builds a FAISS index saved to `wws/faiss_store/`; `03-构建检索.py` loads it from `../faiss_store`
- Batch size for FAISS ingestion is 10
- Most scripts hardcode `base_url="https://api.deepseek.com"` rather than reading from env

## LangChain package fragmentation

Due to LangChain's package split, imports come from several namespaces:
- `langchain_core` — prompts, messages
- `langchain_openai` — ChatOpenAI
- `langchain_community` — embeddings (HuggingFaceEmbeddings), vector stores (FAISS), document loaders (WebBaseLoader)
- `langchain_classic` — chains (create_stuff_documents_chain, create_retrieval_chain)
- `langchain_huggingface` — HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
- `langchain_deepseek` — ChatDeepSeek
- `langchain_text_splitters` — RecursiveCharacterTextSplitter
- `langchain.chat_models` — init_chat_model
