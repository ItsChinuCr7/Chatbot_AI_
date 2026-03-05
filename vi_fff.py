import streamlit as st
import time
import os
import tempfile

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


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
# CUSTOM CSS (GPT STYLE)
# =============================
st.markdown("""
<style>
.chat-container {
    padding: 20px;
}

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

.sidebar .sidebar-content {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)


# =============================
# LOAD MODEL
# =============================
chat = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434"
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional AI document assistant.
Answer ONLY using the context.

Context:
{context}

Question:
{question}

If answer not found, say "I don't know".
"""
)


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
                chunk_size=500,
                chunk_overlap=50
            )

            docs = splitter.split_documents(documents)

            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(INDEX_PATH)

            st.session_state["vector_db"] = vector_db
            st.success("Documents indexed successfully!")


# Auto load saved index
if "vector_db" not in st.session_state and os.path.exists(INDEX_PATH):
    st.session_state["vector_db"] = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =============================
# MAIN CHAT AREA
# =============================
st.title("🤖 AI Document Intelligence Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{message}</div>', unsafe_allow_html=True)


# Chat input (GPT style)
user_input = st.chat_input("Ask something about your documents...")


if user_input:

    if "vector_db" not in st.session_state:
        st.warning("Please process documents first from sidebar.")
    else:

        # Add user message
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):

            retriever = st.session_state["vector_db"].as_retriever(
                search_kwargs={"k": 3}
            )

            relevant_docs = retriever.invoke(user_input)

            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            formatted_prompt = prompt.format(
                context=context,
                question=user_input
            )

            response = chat.invoke(formatted_prompt)

            answer = response.content

        # Add bot response
        st.session_state.chat_history.append(("bot", answer))

        st.rerun()