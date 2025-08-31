
import os
import io
import tempfile
from typing import List, Optional, Iterable, Set, Tuple

import streamlit as st

# LangChain core + OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Loaders, splitters, vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="Advanced RAG (Query Rewrite + Multi-Query + Vector DB)", page_icon="📚💬", layout="wide")
st.title("📚💬 Advanced RAG + General Chat")

# -------------------------
# Sidebar: API & Settings
# -------------------------
with st.sidebar:
    st.header("🔐 API & Endpoints")
    default_key = os.environ.get("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    base_url = st.text_input("OpenAI Base URL (optional)", value=os.environ.get("OPENAI_BASE_URL", ""), help="Leave empty for api.openai.com. For Azure/others, set full base URL.")
    org = st.text_input("OpenAI Org (optional)", value=os.environ.get("OPENAI_ORG_ID", ""))

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if org:
        os.environ["OPENAI_ORG_ID"] = org

    st.header("⚙️ Model & Mode")
    openai_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)
    mode = st.radio("Mode", ["RAG (PDF Q&A)", "General Chat"], index=0)

# Hard-stop if missing API key
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Set your OpenAI API key in the left sidebar to start.")
    st.stop()

# LLM instance shared by both modes
try:
    llm = ChatOpenAI(model=openai_model, temperature=temperature)
except Exception as e:
    st.error(f"Failed to initialize ChatOpenAI: {e}")
    st.stop()

# -------------------------
# Session state
# -------------------------
if "messages_general" not in st.session_state:
    st.session_state.messages_general = []
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "prompt_profile" not in st.session_state:
    st.session_state.prompt_profile = "Default QA"
if "vector_dir" not in st.session_state:
    st.session_state.vector_dir = "faiss_index"  # default location

# -------------------------
# Prompt Catalog
# -------------------------
PROMPT_CATALOG = {
    "Default QA": (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the context is insufficient, say you don't know."
    ),
    "Beginner-friendly": (
        "Explain like I'm new to the topic. Use simple language and short paragraphs. "
        "Answer ONLY from the provided context. If the context is insufficient, say you don't know."
    ),
    "Executive summary": (
        "Give a crisp, executive-ready summary in 4-6 bullets, starting with the key takeaway. "
        "Answer ONLY from the provided context. If the context is insufficient, say you don't know."
    ),
    "Bullet extractor": (
        "Extract the key facts as a bulleted list. Be concise. "
        "Answer ONLY from the provided context. If the context is insufficient, say you don't know."
    ),
    "Code explainer": (
        "Explain the concepts to a developer and include short code snippets ONLY if grounded in the context. "
        "Answer ONLY from the provided context. If the context is insufficient, say you don't know."
    ),
}

# -------------------------
# Persistent Vector Store helpers
# -------------------------
def save_vectorstore(vs: FAISS, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    vs.save_local(folder_path=save_dir)

def load_vectorstore(save_dir: str, embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
    if not os.path.isdir(save_dir):
        return None
    try:
        return FAISS.load_local(folder_path=save_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

# -------------------------
# RAG helpers
# -------------------------
def build_vectorstore_from_pdfs(uploaded_files: List["st.runtime.uploaded_file_manager.UploadedFile"]) -> FAISS:
    \"\"\"Create a FAISS vector store from Streamlit UploadedFile PDFs and return it.\"\"\"
    docs_all = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uf in uploaded_files:
        try:
            pdf_bytes = uf.read()
            if not pdf_bytes:
                uf.seek(0)
                pdf_bytes = uf.read()
            if not pdf_bytes:
                raise ValueError(f"Uploaded file {uf.name} appears empty.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            loaded = loader.load()  # List[Document]
            for d in loaded:
                d.metadata["source_name"] = getattr(uf, "name", os.path.basename(tmp_path))
            chunks = splitter.split_documents(loaded)
            docs_all.extend(chunks)
        except Exception as e:
            raise RuntimeError(f"Failed to ingest {getattr(uf, 'name', 'uploaded file')}: {e}")

    if not docs_all:
        raise RuntimeError("No text extracted from the uploaded PDFs.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs_all, embeddings)
    return vs

def make_retriever(vs: FAISS, k: int = 4):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})

def make_rag_chain_from_docs(docs, prompt_profile: str, question: str, llm: ChatOpenAI):
    system = PROMPT_CATALOG.get(prompt_profile, PROMPT_CATALOG["Default QA"])
    def _format_docs(retrieved_docs):
        return "\\n\\n".join(doc.page_content for doc in retrieved_docs)
    prompt_t = PromptTemplate(
        template=(
            "{system}\\n\\nContext:\\n{context}\\n\\nQuestion: {question}"
        ),
        input_variables=["system", "context", "question"],
    )
    context_text = _format_docs(docs) if docs else ""
    chain = (
        RunnableParallel({
            "system": RunnableLambda(lambda _: system),
            "context": RunnableLambda(lambda _: context_text),
            "question": RunnableLambda(lambda _: question),
        }) | prompt_t | llm | StrOutputParser()
    )
    return chain

# -------------------------
# Query rewriting
# -------------------------
REWRITE_SYSTEM = (
    "Rewrite the user's question into a precise, self-contained query for retrieval over technical PDFs. "
    "Keep domain terms, expand acronyms only when helpful, remove chit-chat, and output ONLY the rewritten query."
)

def rewrite_query(user_q: str, llm: ChatOpenAI) -> str:
    tmpl = PromptTemplate(
        template="System: {sys}\nUser question: {q}\nRewritten query:",
        input_variables=["sys", "q"]
    )
    chain = tmpl | llm | StrOutputParser()
    try:
        return chain.invoke({"sys": REWRITE_SYSTEM, "q": user_q}).strip()
    except Exception:
        return user_q  # fallback

# -------------------------
# Our own Multi-Query Wrapper (version-agnostic)
# -------------------------
class MultiQueryWrapper:
    \"\"\"A minimal retriever wrapper that expands the query via LLM N times and merges results.\"\"\"
    def __init__(self, base_retriever, llm: ChatOpenAI, num_queries: int = 4, include_original: bool = True):
        self.base = base_retriever
        self.llm = llm
        self.num_queries = max(1, int(num_queries))
        self.include_original = include_original

        self.prompt = PromptTemplate(
            template=(
                "You will generate {n} diverse, semantically distinct paraphrases for the user question, "
                "suitable for retrieving passages from technical PDFs. Separate each query on a new line.\n\n"
                "User question: {q}\n\n"
                "Queries:"
            ),
            input_variables=["n", "q"]
        )
        self.chain = self.prompt | llm | StrOutputParser()

    def _expand_queries(self, question: str) -> List[str]:
        try:
            raw = self.chain.invoke({"n": str(self.num_queries), "q": question})
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            # remove bullets or numbering
            cleaned = [ln.lstrip("-•0123456789. ").strip() for ln in lines]
        except Exception:
            cleaned = []
        queries = cleaned[: self.num_queries]
        if self.include_original:
            queries = [question] + queries
        # de-dup
        seen: Set[str] = set()
        unique: List[str] = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        return unique

    def get_relevant_documents(self, question: str):
        queries = self._expand_queries(question)
        docs = []
        seen: Set[Tuple[str, str]] = set()
        for q in queries:
            for d in self.base.get_relevant_documents(q):
                key = (d.page_content, str(d.metadata))
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return docs

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1, 2])

with left:
    if mode == "RAG (PDF Q&A)":
        st.subheader("📄 Knowledge PDFs")
        files = st.file_uploader(
            "Upload one or more PDFs (they become your knowledge base)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        st.text_input("Vector store directory", key="vector_dir", help="Where the FAISS index is stored/loaded from.", value=st.session_state.vector_dir)

        st.markdown("**Prompt profile**")
        prompt_profile = st.selectbox("Choose how answers should be phrased", list(PROMPT_CATALOG.keys()), index=list(PROMPT_CATALOG.keys()).index(st.session_state.prompt_profile) if st.session_state.prompt_profile in PROMPT_CATALOG else 0)

        st.markdown("**Advanced retrieval**")
        enable_rewrite = st.checkbox("Enable LLM Query Rewriting", value=True)
        use_multi_query = st.checkbox("Enable Multi-Query Expansion", value=True)
        num_queries = st.slider("Reformulations", 2, 8, 4)
        top_k = st.slider("Chunks per search (k)", 2, 12, 4)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔧 Build from uploaded PDFs", disabled=not files):
                try:
                    vs = build_vectorstore_from_pdfs(files)
                    save_vectorstore(vs, st.session_state.vector_dir)
                    base_retriever = make_retriever(vs, k=top_k)
                    retriever = base_retriever
                    if use_multi_query:
                        retriever = MultiQueryWrapper(base_retriever, llm=llm, num_queries=int(num_queries), include_original=True)
                    st.session_state.retriever = retriever
                    st.session_state.rag_ready = True
                    st.session_state.prompt_profile = prompt_profile
                    st.session_state["rewrite_toggle"] = enable_rewrite
                    st.success(f"Vector store built and saved to '{st.session_state.vector_dir}' ✅")
                except Exception as e:
                    st.session_state.rag_ready = False
                    st.error(f"Failed to build vector store: {e}")
        with col_b:
            if st.button("📦 Load existing vector store"):
                try:
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vs_loaded = load_vectorstore(st.session_state.vector_dir, embeddings)
                    if not vs_loaded:
                        raise RuntimeError("No existing vector store found or incompatible format.")
                    base_retriever = make_retriever(vs_loaded, k=top_k)
                    retriever = base_retriever
                    if use_multi_query:
                        retriever = MultiQueryWrapper(base_retriever, llm=llm, num_queries=int(num_queries), include_original=True)
                    st.session_state.retriever = retriever
                    st.session_state.rag_ready = True
                    st.session_state.prompt_profile = prompt_profile
                    st.session_state["rewrite_toggle"] = enable_rewrite
                    st.success(f"Loaded vector store from '{st.session_state.vector_dir}' ✅")
                except Exception as e:
                    st.session_state.rag_ready = False
                    st.error(f"Failed to load vector store: {e}")

with right:
    # Display chat history
    history = st.session_state.messages_rag if mode == "RAG (PDF Q&A)" else st.session_state.messages_general
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your message…")

    if prompt:
        # Record user
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if mode == "General Chat":
            with st.chat_message("assistant"):
                try:
                    resp = llm.invoke(prompt)
                    content = resp.content if hasattr(resp, "content") else str(resp)
                except Exception as e:
                    content = f"Error: {e}"
                st.markdown(content)
            history.append({"role": "assistant", "content": content})

        else:  # RAG mode
            with st.chat_message("assistant"):
                if not st.session_state.rag_ready or st.session_state.retriever is None:
                    st.warning("Please build or load a vector store first (left panel).")
                    content = "I need your PDFs vector store to answer from context."
                    st.markdown(content)
                else:
                    try:
                        # 1) LLM-based query rewriting
                        effective_query = prompt
                        if st.session_state.get("rewrite_toggle", True):
                            rewritten = rewrite_query(prompt, llm)
                            if rewritten and rewritten.strip() and rewritten.strip().lower() != prompt.strip().lower():
                                effective_query = rewritten
                                st.caption(f"📝 Rewritten query: {effective_query}")

                        # 2) Retrieve from vector DB (wrapper may expand to multiple queries)
                        retr = st.session_state.retriever
                        docs = retr.get_relevant_documents(effective_query) if retr else []

                        # 3) Answer strictly from retrieved context
                        chain = make_rag_chain_from_docs(docs, st.session_state.prompt_profile, effective_query, llm)
                        content = chain.invoke({})
                        st.markdown(content)

                        # Citations
                        if docs:
                            seen = set()
                            cites = []
                            for d in docs[:8]:
                                name = d.metadata.get("source_name") or d.metadata.get("source") or "PDF"
                                label = name
                                page = d.metadata.get("page")
                                if page is not None:
                                    label += f" (page {page})"
                                if label not in seen:
                                    seen.add(label)
                                    cites.append(label)
                            if cites:
                                st.info("**Sources:**\n- " + "\n- ".join(cites))

                    except Exception as e:
                        content = f"RAG error: {e}"
                        st.markdown(content)
            history.append({"role": "assistant", "content": content})

    # Persist
    if mode == "RAG (PDF Q&A)":
        st.session_state.messages_rag = history
    else:
        st.session_state.messages_general = history
