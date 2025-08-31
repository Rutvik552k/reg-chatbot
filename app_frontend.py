import os
import uuid
import tempfile
from typing import List, Dict

import streamlit as st
from langchain_openai import OpenAIEmbeddings

from db import init_db, upsert_thread, insert_message, load_threads, load_messages_for_user, is_thread_owned_by
from auth import auth_gate
from rag_core import get_llm, build_retriever_from_paths, make_rag_chain, history_string
from vectorstore import user_thread_index_dir, save_faiss,load_faiss
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG (Modular) — Auth + Threads", page_icon="🧩", layout="wide")
st.title("🧩 Modular RAG — Login + Threads + Per-Account Isolation")

# -------------------------
# Init / Auth
# -------------------------
init_db()
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    auth_gate()
    st.stop()

with st.sidebar:
    st.header(f"👤 {st.session_state.user['email']}")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

def _new_tid() -> str:
    return str(uuid.uuid4())[:8]

# -------------------------
# Load threads restricted to the current user
# -------------------------
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "order" not in st.session_state:
    st.session_state.order = []
if "current" not in st.session_state:
    db_threads = load_threads(st.session_state.user["id"])
    if db_threads:
        for t in db_threads:
            tid = t["id"]
            st.session_state.threads[tid] = {
                "title": t["title"],
                "index_dir": t["index_dir"],
                "messages": load_messages_for_user(st.session_state.user["id"], tid),
                "retriever": None,
            }
        st.session_state.order = [t["id"] for t in db_threads]
        st.session_state.current = st.session_state.order[0]
    else:
        tid = _new_tid()
        title = "New chat"
        idx_dir = user_thread_index_dir(st.session_state.user["id"], tid)
        upsert_thread(st.session_state.user["id"], tid, title, idx_dir)
        st.session_state.threads[tid] = {"title": title, "index_dir": idx_dir, "messages": [], "retriever": None}
        st.session_state.order = [tid]
        st.session_state.current = tid

# Ensure current thread belongs to the current user
if not is_thread_owned_by(st.session_state.user["id"], st.session_state.current):
    # pick a safe thread or create a new one
    user_threads = load_threads(st.session_state.user["id"])
    if user_threads:
        st.session_state.current = user_threads[0]["id"]
    else:
        tid = _new_tid()
        title = "New chat"
        idx_dir = user_thread_index_dir(st.session_state.user["id"], tid)
        upsert_thread(st.session_state.user["id"], tid, title, idx_dir)
        st.session_state.threads[tid] = {"title": title, "index_dir": idx_dir, "messages": [], "retriever": None}
        st.session_state.order = [tid]
        st.session_state.current = tid

# -------------------------
# Sidebar: thread list (only current user's)
# -------------------------
with st.sidebar:
    st.header("🧵 Your chats")
    db_threads = load_threads(st.session_state.user["id"])
    if db_threads:
        for t in db_threads:
            tid = t["id"]
            if tid in st.session_state.threads:
                st.session_state.threads[tid]["title"] = t["title"]
                st.session_state.threads[tid]["index_dir"] = t["index_dir"]
                # refresh messages securely
                st.session_state.threads[tid]["messages"] = load_messages_for_user(st.session_state.user["id"], tid)
            else:
                st.session_state.threads[tid] = {
                    "title": t["title"],
                    "index_dir": t["index_dir"],
                    "messages": load_messages_for_user(st.session_state.user["id"], tid),
                    "retriever": None,
                }
        st.session_state.order = [t["id"] for t in db_threads]

    labels = [f"{st.session_state.threads[tid]['title']} — {tid}" for tid in st.session_state.order[:30]]
    if labels:
        idx = st.selectbox("Switch chat", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)
        chosen = st.session_state.order[idx]
        # Verify ownership on switch
        if chosen != st.session_state.current and is_thread_owned_by(st.session_state.user["id"], chosen):
            st.session_state.current = chosen

    if st.button("➕ New chat"):
        tid = _new_tid()
        title = "New chat"
        idx_dir = user_thread_index_dir(st.session_state.user["id"], tid)
        upsert_thread(st.session_state.user["id"], tid, title, idx_dir)
        st.session_state.threads[tid] = {"title": title, "index_dir": idx_dir, "messages": [], "retriever": None}
        st.session_state.order.insert(0, tid)
        st.session_state.current = tid

# -------------------------
# OpenAI settings
# -------------------------
with st.sidebar:
    st.header("🔐 OpenAI")
    # default_key = os.environ.get("OPENAI_API_KEY", "")
    # api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    # if api_key:
    #     os.environ["OPENAI_API_KEY"] = api_key
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Set your OpenAI API key in the sidebar to start.")
    st.stop()

llm = get_llm(model=model, temperature=temperature)

# -------------------------
# Current thread (ownership enforced)
# -------------------------
cur = st.session_state.current
if not is_thread_owned_by(st.session_state.user["id"], cur):
    st.error("Access denied for this thread.")
    st.stop()

thread = st.session_state.threads.get(cur)

st.subheader(f"Thread: {thread['title']}")
new_title = st.text_input("Rename this chat", value=thread["title"])
if new_title != thread["title"]:
    thread["title"] = new_title
    upsert_thread(st.session_state.user["id"], cur, new_title, thread["index_dir"])

st.markdown("**Upload PDFs to build or refresh your RAG index (saved per thread).**")
upl = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader_{cur}")
if st.button("🔧 Build/Refresh Index", disabled=not upl):
    try:
        paths: List[str] = []
        for uf in upl:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                paths.append(tmp.name)
        retriever, vs = build_retriever_from_paths(paths)
        save_faiss(vs, thread["index_dir"])
        thread["retriever"] = retriever
        st.success("Index built & saved for this thread ✅")
    except Exception as e:
        thread["retriever"] = None
        st.error(f"Failed to build index: {e}")

st.caption(f"Index location: `{thread['index_dir']}` (auto-loads if present)")

# Auto-load vector store
if thread["retriever"] is None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs_loaded = load_faiss(thread["index_dir"], embeddings)
    if vs_loaded:
        thread["retriever"] = vs_loaded.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -------------------------
# Chat UI
# -------------------------
for m in thread["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask about your PDFs...")
if user_msg:
    if not is_thread_owned_by(st.session_state.user["id"], cur):
        st.error("Access denied for this thread.")
        st.stop()

    thread["messages"].append({"role": "user", "content": user_msg})
    insert_message(cur, "user", user_msg)
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        if thread["retriever"] is None:
            content = "⚠️ Please upload PDFs and click **Build/Refresh Index** first."
            st.markdown(content)
            thread["messages"].append({"role": "assistant", "content": content})
            insert_message(cur, "assistant", content)
        else:
            docs = thread["retriever"].get_relevant_documents(user_msg)
            if not docs:
                content = "⚠️ No relevant content found in the vector store for your query."
                st.markdown(content)
                thread["messages"].append({"role": "assistant", "content": content})
                insert_message(cur, "assistant", content)
            else:
                preview = "\n\n---\n\n".join(d.page_content[:500] for d in docs[:2])
                with st.expander("🔍 Retrieved Chunks Preview"):
                    st.write(preview)

                hist = history_string(thread["messages"])
                chain = make_rag_chain(thread["retriever"], llm, history_text=hist)

                def _stream():
                    try:
                        for chunk in chain.stream(user_msg):
                            if isinstance(chunk, str):
                                yield chunk
                            else:
                                yield getattr(chunk, "content", "") or ""
                    except Exception as e:
                        yield f"\n[stream error: {e}]"

                full = st.write_stream(_stream())
                thread["messages"].append({"role": "assistant", "content": full})
                insert_message(cur, "assistant", full)

# Keep order
if cur in st.session_state.order:
    st.session_state.order.remove(cur)
st.session_state.order.insert(0, cur)
