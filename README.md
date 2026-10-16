# 🤖 Enterprise RAG & Agentic Knowledge Assistant

An advanced, end-to-end Retrieval-Augmented Generation (RAG) and Agentic workflow system designed to solve enterprise information overload, eliminate LLM hallucinations, and reduce data retrieval latency. Built using **Python, LangChain, LangGraph, FastAPI, ChromaDB, and Streamlit**.

---

## 🚀 Key Features

* **Context-Aware Document Ingestion & RAG**: Automatically ingests enterprise documents (PDFs, text files), splits them into optimized chunks, and stores dense vector embeddings in **ChromaDB** for hyper-fast semantic search.
* **Agentic Workflows (LangGraph)**: Leverages stateful multi-step graphs to handle complex user intents, route queries dynamically, and execute targeted tool calls.
* **Source-Grounded Answers**: Combines retrieved context with LLM APIs to ensure responses are strictly grounded in internal documentation, eliminating hallucinations.
* **High-Performance Backend & UI**: Powered by a robust FastAPI backend and an interactive, user-friendly **Streamlit** frontend chat interface.

---

## 🛠️ Tech Stack

* **Orchestration & Frameworks**: LangChain, LangGraph
* **Vector Database**: ChromaDB
* **API & Backend**: FastAPI, REST APIs
* **Frontend**: Streamlit
* **Language**: Python 3.10+

---

## 📂 Project Architecture

```text
enterprise-rag-assistant/
│
├── app.py                  # Streamlit Frontend UI
├── backend/
│   ├── main.py             # FastAPI application endpoints
│   ├── rag_pipeline.py     # Document loader, text splitter, and retriever chain
│   └── agent.py            # LangGraph state machine & tool-calling logic
├── data/                   # Sample enterprise documents / knowledge base
├── requirements.txt        # Project dependencies
└── README.md
