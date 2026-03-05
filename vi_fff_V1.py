import streamlit as st
import time
import os
import tempfile

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

INDEX_PATH = "faiss_index"


# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>

.user-bubble {
    background-color: #2b313e;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    text-align: right;
    color: white;
}

.bot-bubble {
    background-color: #1e1e1e;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: #00ffcc;
}

</style>
""", unsafe_allow_html=True)


# =============================
# LOAD MODELS (CACHED)
# =============================
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="phi3",   # much faster than llama3
        base_url="http://localhost:11434"
    )


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"   # fast + strong retrieval
    )


chat = load_llm()
embeddings = load_embeddings()


# =============================
# RLM STYLE RECURSIVE REASONING
# =============================
def recursive_reasoning(query, retriever, depth=0, max_depth=1):

    docs = retriever.invoke(query)

    context = "\n".join(d.page_content[:600] for d in docs)

    prompt = f"""
You are an AI assistant exploring documents.

Question:
{query}

Context:
{context}

If the answer is found return:

FINAL_ANSWER: <answer>

If not enough information return:

SEARCH_QUERY: <better query>
"""

    response = chat.invoke(prompt).content.strip()

    if "FINAL_ANSWER:" in response or depth >= max_depth:
        return response.replace("FINAL_ANSWER:", "").strip()

    if "SEARCH_QUERY:" in response:
        new_query = response.split("SEARCH_QUERY:")[1].strip()
        return recursive_reasoning(new_query, retriever, depth + 1)

    return response


# =============================
# SIDEBAR
# =============================
with st.sidebar:

    st.title("⚙ Document Settings")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("📂 Process Documents"):

        if not uploaded_files:
            st.warning("Upload PDFs first.")
        else:

            documents = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.read())

                    loader = PyPDFLoader(tmp.name)
                    documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            docs = splitter.split_documents(documents)

            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(INDEX_PATH)

            st.session_state["vector_db"] = vector_db

            st.success("Documents indexed successfully!")


# =============================
# AUTO LOAD INDEX
# =============================
if "vector_db" not in st.session_state and os.path.exists(INDEX_PATH):

    st.session_state["vector_db"] = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =============================
# MAIN UI
# =============================
st.title("🤖 AI Document Intelligence Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =============================
# DISPLAY CHAT
# =============================
for role, message in st.session_state.chat_history:

    if role == "user":
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="bot-bubble">{message}</div>', unsafe_allow_html=True)


# =============================
# CHAT INPUT
# =============================
user_input = st.chat_input("Ask something about your documents...")


if user_input:

    if "vector_db" not in st.session_state:
        st.warning("Please process documents first from sidebar.")

    else:

        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):

            retriever = st.session_state["vector_db"].as_retriever(
                search_kwargs={"k": 5}
            )

            start = time.time()

            answer = recursive_reasoning(user_input, retriever)

            end = time.time()

        st.session_state.chat_history.append(("bot", answer))

        st.caption(f"Response time: {round(end-start,2)} sec")

        st.rerun()