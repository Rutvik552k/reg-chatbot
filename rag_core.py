
import os
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)

def build_retriever_from_paths(pdf_paths: List[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_all = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()  # List[Document]
        # text =" ".join(doc.page_content for doc in pages)
        chunks = splitter.split_documents(pages)
        for d in chunks:
            d.metadata["source_name"] = os.path.basename(path)
        docs_all.extend(chunks)
    if not docs_all:
        raise RuntimeError("No text extracted from PDFs.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs_all, embeddings)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 4}), vs

def make_rag_chain(retriever, llm: ChatOpenAI, history_text: str = ""):
    def format_docs(retrieved_docs):
        return "\\n\\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the provided context to answer as best as possible.\\n"
            "If no context is provided, say you don't know.\\n\\n"
            "Conversation history (may be empty):\\n{history}\\n\\n"
            "Context:\\n{context}\\n\\n"
            "Question: {question}"
        ),
        input_variables=["history", "context", "question"],
    )

    parallel = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs),
         "history": RunnableLambda(lambda _: history_text),
         "question": RunnablePassthrough()}
    )
    return parallel | prompt | llm | StrOutputParser()

def history_string(msgs: List[Dict[str, str]], N: int = 6) -> str:
    recent = msgs[-(2*N):]
    return "".join(f"{'User' if mm['role']=='user' else 'Assistant'}: {mm['content']}\\n" for mm in recent)
