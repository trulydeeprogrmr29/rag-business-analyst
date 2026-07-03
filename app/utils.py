import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def ingest_and_embed(uploaded_files):
    os.makedirs(DATA_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

    # Clear cache so load_vectorstore reloads fresh
    load_vectorstore.clear()

    return vectorstore, len(all_chunks)

def get_retriever(vectorstore, k=6):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,               
            "fetch_k": 20         # ← fetch 20 candidates, pick 6 diverse ones
        }
    )

def build_prompt(question: str, context_chunks: list) -> str:
    context = ""
    for i, doc in enumerate(context_chunks):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context += f"\n[Source {i+1}: {source} | Page {page}]\n"
        context += doc.page_content + "\n"

    return f"""You are a professional business analyst assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Always:
- Cite your sources (which document and page)
- Explain your reasoning step by step
- Be concise but complete

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def ask(question: str, retriever) -> dict:
    chunks = retriever.invoke(question)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a professional business analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    sources = [
        {
            "source": doc.metadata.get("source", "").replace("data\\", "").replace("data/", ""),
            "page": doc.metadata.get("page")
        }
        for doc in chunks
    ]

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": sources
    }

def list_documents():
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]