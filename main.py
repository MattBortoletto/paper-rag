import os 
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_docs(path):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {path}")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = splitter.split_documents(docs)
    print(f"Splitted docs into {len(splits)} chunks")
    return splits 

def make_rag_chain(vector_db, llm_name, context_length):
    llm = ChatOllama(
        model=llm_name,
        temperature=0,
        num_ctx=context_length
    )
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    template = """You are a helpful, accurate assistant for question answering over provided context.

Use ONLY the information in the context below to answer the question.
If the answer is not contained in the context, say: "I don't know based on the provided context."

Keep your answer:
- Concise and to the point
- In the same language as the question
- Well-structured (use bullet points or short paragraphs where helpful)

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


if __name__ == "__main__":

    if not os.path.exists("papers"):
        raise FileNotFoundError("papers/ directory not found!")
    chroma_dir = "chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)
    embedding_model_name = "all-minilm"
    llm_name = "qwen3:0.6b"
    context_length = 8192
    
    docs = load_docs("papers")
    chunks = split_docs(docs) 
    embedding_func = OllamaEmbeddings(model=embedding_model_name)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_func,
        persist_directory=chroma_dir
    )
    # TODO load an existing DB: vector_db = get_vector_store(embedding_func)
    rag_chain = make_rag_chain(vector_db, llm_name, context_length)

    question = "What is ProToM?"
    response = rag_chain.invoke(question)
    print(response)
    
    breakpoint()
