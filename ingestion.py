import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = './data'
FAISS_INDEX_DIR = './vector_db'

def ingest_documents():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f'Created {DATA_DIR}/ folder. Please add at least one PDF file inside it.')
        return

    print('Loading PDF documents from data/ directory...')
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print('No PDF files found in data/. Please add a PDF and run again.')
        return

    print(f'Loaded {len(documents)} page(s) successfully.')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split documents into {len(chunks)} chunks.')

    print('Loading free local embedding model (all-MiniLM-L6-v2)...')
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    print('Generating embeddings and saving vector index locally...')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print(f'Success! Vector database saved locally in "{FAISS_INDEX_DIR}".')

if __name__ == '__main__':
    ingest_documents()
