import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

FAISS_INDEX_DIR = "./vector_db"

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str

def retrieve_node(state: AgentState):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_DIR):
        vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(state["question"], k=4)
        return {"context": [doc.page_content for doc in docs]}
    return {"context": []}

def llm_node(state: AgentState):
    # Ensures API key is present from environment
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Google API Key is missing. Please set it in your .env file.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    context_str = "\n\n".join(state["context"]) if state["context"] else "No specific document context retrieved."
    
    prompt = f"""You are Klaus, an intelligent knowledge assistant. Answer the user's question clearly and accurately based on the context.

Context:
{context_str}

User Question: {state['question']}

Answer clearly using Markdown formatting (use bullet points or numbered lists if requested):"""
    
    response = llm.invoke(prompt)
    res_text = response.content
    if isinstance(res_text, list) and len(res_text) > 0 and isinstance(res_text[0], dict):
        res_text = res_text[0].get("text", str(res_text))
    return {"answer": str(res_text)}

def get_workflow_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("llm", llm_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "llm")
    workflow.add_edge("llm", END)
    return workflow.compile()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klaus - AI Knowledge Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        .markdown-body ul { list-style-type: disc; margin-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
        .markdown-body ol { list-style-type: decimal; margin-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
        .markdown-body p { margin-bottom: 0.5rem; }
        .markdown-body strong { color: #f4f4f5; font-weight: 600; }
    </style>
</head>
<body class="bg-black text-gray-100 font-sans min-h-screen flex flex-col">
    <!-- Header -->
    <header class="border-b border-zinc-800 p-4 flex justify-between items-center max-w-4xl w-full mx-auto">
        <div class="flex items-center space-x-2">
            <span class="text-xl font-bold tracking-tight text-white">⚡ Klaus</span>
            <span class="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full">Enterprise RAG</span>
        </div>
    </header>   
    

    <!-- Main Container -->
    <main class="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col space-y-6">
        <!-- Upload Section -->
        <div class="bg-zinc-950 border border-zinc-800 rounded-2xl p-4">
            <h2 class="text-sm font-semibold text-zinc-300 mb-2">Upload Knowledge Base Document</h2>
            <div class="flex items-center space-x-3">
                <input type="file" id="pdfFile" accept=".pdf" class="block w-full text-sm text-zinc-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-xl file:border-0
                    file:text-sm file:font-semibold
                    file:bg-zinc-800 file:text-zinc-200
                    hover:file:bg-zinc-700 cursor-pointer">
                <button onclick="uploadPdf()" class="bg-white hover:bg-zinc-200 text-black text-sm font-semibold px-4 py-2 rounded-xl transition whitespace-nowrap">Process PDF</button>
            </div>
            <p id="uploadStatus" class="text-xs mt-2 text-zinc-500"></p>
        </div>

        <!-- Chat History Area -->
        <div id="chatContainer" class="flex-1 space-y-4 overflow-y-auto max-h-[55vh] pr-2">
            <div class="bg-zinc-900/50 border border-zinc-800/60 rounded-2xl p-4 text-zinc-400 text-sm">
                👋 Hello! I am Klaus, your enterprise assistant. Ask me anything about your uploaded documents.
            </div>
        </div>

        <!-- Query Input Box -->
        <div class="sticky bottom-4 bg-zinc-950 border border-zinc-800 rounded-2xl p-3 flex items-center space-x-3">
            <input type="text" id="queryInput" placeholder="Ask anything about your documents..." 
                   class="flex-1 bg-transparent border-none focus:outline-none text-sm text-zinc-200 placeholder-zinc-500 px-2"
                   onkeypress="if(event.key === 'Enter') runQuery()">
            <button onclick="runQuery()" class="bg-white hover:bg-zinc-200 text-black text-sm font-semibold px-4 py-2 rounded-xl transition">Ask Klaus</button>
        </div>
    </main>

    <script>
        async function uploadPdf() {
            const fileInput = document.getElementById("pdfFile");
            const status = document.getElementById("uploadStatus");
            if(fileInput.files.length === 0) {
                alert("Please select a PDF file first.");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            status.innerText = "Processing and indexing document chunks...";
            try {
                let response = await fetch("/upload", { method: "POST", body: formData });
                let result = await response.json();
                status.innerText = result.message || "Document successfully indexed!";
            } catch (e) {
                status.innerText = "Error uploading document.";
            }
        }

        async function runQuery() {
            const queryInput = document.getElementById("queryInput");
            const chatContainer = document.getElementById("chatContainer");
            const query = queryInput.value.trim();

            if(!query) return;

            // Append User Message
            chatContainer.innerHTML += `
                <div class="flex justify-end">
                    <div class="bg-zinc-800 text-zinc-100 rounded-2xl px-4 py-3 max-w-lg text-sm">${query}</div>
                </div>
            `;
            queryInput.value = "";
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Loading state
            const loadingId = "loading-" + Date.now();
            chatContainer.innerHTML += `
                <div id="${loadingId}" class="flex justify-start">
                    <div class="bg-zinc-900 border border-zinc-800 text-zinc-400 rounded-2xl px-4 py-3 text-sm animate-pulse">Klaus is thinking...</div>
                </div>
            `;
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const formData = new FormData();
            formData.append("question", query);

            try {
                let response = await fetch("/query", { method: "POST", body: formData });
                let result = await response.json();
                
                const loadingElem = document.getElementById(loadingId);
                if(loadingElem) loadingElem.remove();

                const formattedResponse = marked.parse(result.answer);

                chatContainer.innerHTML += `
                    <div class="flex justify-start">
                        <div class="bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-2xl px-4 py-3 max-w-2xl text-sm leading-relaxed markdown-body">
                            ${formattedResponse}
                        </div>
                    </div>
                `;
            } catch (err) {
                const loadingElem = document.getElementById(loadingId);
                if(loadingElem) loadingElem.remove();
                
                chatContainer.innerHTML += `
                    <div class="flex justify-start">
                        <div class="bg-red-950/50 border border-red-900 text-red-300 rounded-2xl px-4 py-3 text-sm">Error generating response. Check server connection.</div>
                    </div>
                `;
            }
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    os.remove(tmp_path)
    return {"message": "Document successfully indexed into Klaus's vector store!"}

@app.post("/query")
async def query_agent(question: str = Form(...)):
    try:
        workflow_app = get_workflow_app()
        result = workflow_app.invoke({"question": question, "context": [], "answer": ""})
        return {"answer": result["answer"]}
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)