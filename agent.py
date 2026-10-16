import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

FAISS_INDEX_DIR = "./vector_db"

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str

def retrieve_node(state: AgentState):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"context": [doc.page_content for doc in docs]}

def llm_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    context_str = "\n\n".join(state["context"])
    prompt = f"""Based strictly on the following context, answer the user's question.

Context:
{context_str}

Question: {state["question"]}

Answer:"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("llm", llm_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "llm")
workflow.add_edge("llm", END)
app = workflow.compile()

if __name__ == "__main__":
    query = "What are the main technical skills and hackathon achievements?"
    print(f"Running Agent Query: \"{query}\"\n")
    result = app.invoke({"question": query, "context": [], "answer": ""})
    print("--- Agent Answer ---")
    print(result["answer"])
