import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH

def ingest_from_folder(folder_path: str = "data/pdf"):
    pdf_files = list(Path(folder_path).glob("**/*.pdf"))  # recursive search

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{folder_path}'")
        return 0

    print(f"📂 Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"   - {f.name}")

    # Load all PDFs
    docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            docs.extend(loaded)
            print(f"   ✅ Loaded: {pdf_path.name} ({len(loaded)} pages)")
        except Exception as e:
            print(f"   ❌ Failed: {pdf_path.name} → {e}")

    if not docs:
        print("No documents loaded.")
        return 0

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"\n🔪 Split into {len(chunks)} chunks")



    
    # Embed & store in ChromaDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")
    return len(chunks)


if __name__ == "__main__":
    ingest_from_folder("data/pdf")