from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_DIR = './vector_db'

def query_vector_db(query: str, top_k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs

if __name__ == '__main__':
    test_query = 'What are the main skills and experience?'
    results = query_vector_db(test_query)
    print(f'\n--- Retrieved {len(results)} Chunks for Query: "{test_query}" ---\n')
    for i, doc in enumerate(results, 1):
        print(f'Chunk {i}:\n{doc.page_content}\n' + '-'*40)
