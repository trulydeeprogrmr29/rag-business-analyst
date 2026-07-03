import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
import tempfile

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def get_session_data_dir():
    """
    WHY tempfile: creates a unique folder per session
    so users never see each other's uploaded files
    """
    if "session_data_dir" not in st.session_state:
        st.session_state.session_data_dir = tempfile.mkdtemp()
    return st.session_state.session_data_dir

def get_session_vectorstore_dir():
    if "session_vectorstore_dir" not in st.session_state:
        st.session_state.session_vectorstore_dir = tempfile.mkdtemp()
    return st.session_state.session_vectorstore_dir

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    # On cloud, no pre-existing vectorstore — return None
    vectorstore_dir = st.session_state.get("session_vectorstore_dir")
    if not vectorstore_dir or not os.path.exists(vectorstore_dir):
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings
    )

def ingest_and_embed(uploaded_files):
    data_dir = get_session_data_dir()
    vectorstore_dir = get_session_vectorstore_dir()

    # Save uploaded files to session temp folder
    for uploaded_file in uploaded_files:
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    # Clear old vectorstore
    if os.path.exists(vectorstore_dir):
        shutil.rmtree(vectorstore_dir)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=vectorstore_dir
    )
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
    if "session_data_dir" not in st.session_state:
        return []
    data_dir = st.session_state.session_data_dir
    if not os.path.exists(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if f.endswith(".pdf")]