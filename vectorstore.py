
import os
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_ROOT = "indexes"

def user_thread_index_dir(user_id: str, thread_id: str) -> str:
    path = os.path.join(INDEX_ROOT, user_id, thread_id)
    os.makedirs(path, exist_ok=True)
    return path

def save_faiss(vs: FAISS, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    vs.save_local(folder_path=folder)

def load_faiss(folder: str, embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
    if not os.path.isdir(folder):
        return None
    try:
        return FAISS.load_local(folder_path=folder, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None
