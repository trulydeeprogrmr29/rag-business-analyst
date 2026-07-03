import os
from dotenv import load_dotenv
from groq import Groq
from retriever import load_vectorstore, get_retriever

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question: str, context_chunks: list) -> str:
    context = ""
    for i, doc in enumerate(context_chunks):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context += f"\n[Source {i+1}: {source} | Page {page}]\n"
        context += doc.page_content + "\n"

    prompt = f"""You are a professional business analyst assistant.
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
    return prompt


def ask(question: str, retriever) -> dict:
    # Step 1 - Retrieve
    chunks = retriever.invoke(question)

    # Step 2 - Build prompt
    prompt = build_prompt(question, chunks)

    # Step 3 - Ask Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   
        # WHY this model:
        # - 70B parameters = strong reasoning
        # - Free tier on Groq
        # - Fast inference (Groq's specialty)
        messages=[
            {"role": "system", "content": "You are a professional business analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    # Step 4 - Collect sources
    sources = []
    for doc in chunks:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": sources
    }
if __name__ == "__main__":
    vectorstore = load_vectorstore("vectorstore")
    retriever = get_retriever(vectorstore, k=4)

    question = "What was Apple's total revenue in 2024?"
    result = ask(question, retriever)

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for s in result['sources']:
        print(f"  - {s['source']} | Page {s['page']}")



