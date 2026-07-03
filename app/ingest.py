import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  

load_dotenv()

def load_and_chunk_pdfs(data_folder :str):
    """
    Load all PDFs from a folder and splits them into chunks.
    """
    all_chunks=[]
    #Loop through every file in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            print(f"Loading {file_path}")
            #step 1 Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()  # each page=on document
            # step 2 chunk it
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150,
                length_function=len,
                )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

            print(f"-> {len(pages)} pages | {len(chunks)} chunks created")
    print(f"Total chunks ready: {len(all_chunks)}")
    return all_chunks
# Test it
if __name__ == "__main__":
    chunks = load_and_chunk_pdfs("data")

    # Preview the first chunk
    if chunks:
        print("\n--Sample chunk--")
        print(chunks[0].page_content)
        print("\n--Metadata--")
        print(chunks[0].metadata)
    else:
        print("No chunks created.")
