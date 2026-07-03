import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_vectorstore(persist_directory: str):
    """
    Loads the already-saved vectorstore from disk.
    WHY: We embedded once and saved — now we just load, no re-embedding.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"Vectorstore loaded with {vectorstore._collection.count()} chunks")
    return vectorstore


def get_retriever(vectorstore, k: int = 4):
    """
    Converts vectorstore into a retriever.
    k = number of chunks to retrieve per question
    WHY k=6: enough context without overwhelming the LLM
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",   # cosine similarity search
        search_kwargs={"k": k,'fetch_k': 20}  # fetch 20 candidates, pick 6 diverse ones
    )
    return retriever


# Test it
if __name__ == "__main__":
    vectorstore = load_vectorstore("vectorstore")
    retriever = get_retriever(vectorstore, k=4)

    # Ask a test question
    question = "What was Apple's total revenue in 2024?"
    results = retriever.invoke(question)

    print(f"\n--- Top {len(results)} chunks for: '{question}' ---\n")
    for i, doc in enumerate(results):
        print(f"Chunk {i+1} | Source: {doc.metadata['source']} | Page: {doc.metadata['page']}")
        print(doc.page_content[:200])  # preview first 200 chars
        print()