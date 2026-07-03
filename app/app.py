import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import (
    load_vectorstore,
    ingest_and_embed,
    get_retriever,
    ask,
    list_documents
)

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="RAG Business Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Business Analyst")
st.caption("Ask questions about your financial documents — with sources.")

# ── Session state ─────────────────────────────────
# WHY session_state: Streamlit reruns entire script on each interaction
# session_state persists data across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.header("Document Manager")

    # Show loaded documents
    docs = list_documents()
    if docs:
        st.success(f"{len(docs)} document(s) loaded")
        for doc in docs:
            st.write(f" {doc}")
    else:
        st.warning("No documents loaded yet")

    st.divider()

    # Upload new PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF Reports",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(" Embed Documents", type="primary"):
            with st.spinner("Embedding documents... this may take a minute."):
                vectorstore, total_chunks = ingest_and_embed(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.success(f"{total_chunks} chunks embedded!")
                st.rerun()

    st.divider()

    # Clear chat button
    if st.button(" Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Chat Area ────────────────────────────────
if st.session_state.vectorstore is None:
    st.info(" Upload and embed documents from the sidebar to get started.")
else:
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant", avatar="📊"):
            st.write(chat["answer"])

            # Source highlights
            with st.expander("📎 Sources used"):
                for s in chat["sources"]:
                    st.markdown(f"- **{s['source']}** | Page {s['page']}")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("Analyzing documents..."):
                retriever = get_retriever(st.session_state.vectorstore)
                result = ask(question, retriever)

            st.write(result["answer"])

            with st.expander("📎 Sources used"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}** | Page {s['page']}")

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })

