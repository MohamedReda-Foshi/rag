from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import CHROMA_PATH, TOP_K

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Read the context carefully and answer the question.
Extract and summarize the relevant information directly from the text.
Do not say "the context does not address" — find what IS there and use it.

Context:
{context}

Question: {question}

Answer based on what you found in the context:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K}
    )
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.2)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

_chain = None
_retriever = None

def get_chain():
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()
    return _chain, _retriever

def query(question: str) -> dict:
    chain, retriever = get_chain()
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = [
        {"source": doc.metadata.get("source", "unknown"),
         "page": doc.metadata.get("page", 0)}
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    while True:
        question = input("\n❓ Ask a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        result = query(question)
        print(f"\n💬 Answer: {result['answer']}")
        print(f"\n📄 Sources:")
        for s in result["sources"]:
            print(f"   - {s['source']} (page {s['page']})")