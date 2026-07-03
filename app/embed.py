import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from ingest import load_and_chunk_pdfs

load_dotenv()

def create_vectorstore(chunks, persist_directory :str):
    """
    Takes chunks ->create embeddings ->storees in chromaDB
    """
   
    print("\nInitializing embedding model..")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Step 2 — Create vectorstore and embed all chunks
    # WHY: Chroma stores vectors locally so we don't re-embed every time
    print("\nEmbedding chunks and saving to vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"\n Vectorstore created with {vectorstore._collection.count()} chunks")
    print(f"Saved to: {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs("data")
    create_vectorstore(chunks,persist_directory="vectorstore")