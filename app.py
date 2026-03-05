import streamlit as st
import time

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate


# -----------------------------
# Load Ollama Model
# -----------------------------
chat = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)


# -----------------------------
# Prompt Template
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant who answers questions only based on the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say "I don't know".
"""
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 PDF Chatbot (Ollama + LangChain)")

question = st.text_input("Ask a question from PDFs")


# -----------------------------
# LOAD PDF BUTTON
# -----------------------------
if st.button("📂 Load PDFs"):

    start_time = time.time()

    with st.spinner("Loading PDFs..."):

        loader = DirectoryLoader(
            path="D:/Chatbot_AI/test",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()

        context = "\n".join(
            [doc.page_content for doc in documents]
        )

        st.session_state["context"] = context

    end_time = time.time()
    load_time = round(end_time - start_time, 2)

    st.success(f"PDFs Loaded Successfully in {load_time} seconds")


# -----------------------------
# ASK QUESTION BUTTON
# -----------------------------
if st.button("💬 Submit Question"):

    if "context" not in st.session_state:
        st.warning("Please load PDFs first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        start_time = time.time()

        with st.spinner("Thinking..."):

            formatted_prompt = prompt.format(
                context=st.session_state["context"],
                question=question
            )

            response = chat.invoke(formatted_prompt)

        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        st.subheader("Answer")
        st.write(response.content)
        st.info(f"Response generated in {response_time} seconds")
