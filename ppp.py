import streamlit as st
import time

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =============================
# LOAD OLLAMA MODEL (LOCAL LLM)
# =============================
chat = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434"
)


# =============================
# LOCAL EMBEDDING MODEL
# =============================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =============================
# PROMPT TEMPLATE
# =============================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document analysis assistant.

Step 1: Identify relevant facts from context.
Step 2: Answer only using those facts.
Step 3: Keep answer concise.

Context:
{context}

Question:
{question}

If answer not present, say "I don't know".
"""
)


# =============================
# STREAMLIT UI
# =============================
st.title("📄 Fast PDF Chatbot (Ollama + FAISS)")

question = st.text_input("Ask a question")


# =============================
# DOCUMENT TRANSFORMATION
# =============================
def transform_documents(docs):
    cleaned_docs = []
    for doc in docs:
        text = doc.page_content

        # safer normalization
        text = text.replace("\n\n", " ")
        text = " ".join(text.split())

        doc.page_content = text
        cleaned_docs.append(doc)

    return cleaned_docs


# =============================
# LOAD & BUILD VECTOR DATABASE
# =============================
@st.cache_resource
def load_vector_db():

    loader = DirectoryLoader(
        path="D:/Chatbot_AI/test",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    documents = transform_documents(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(docs, embeddings)

    return vector_db


# =============================
# LOAD PDF BUTTON
# =============================
if st.button("📂 Load PDFs"):

    with st.spinner("Processing PDFs..."):
        st.session_state["vector_db"] = load_vector_db()

    st.success("PDFs processed and indexed!")


# =============================
# ASK QUESTION
# =============================
if st.button("💬 Submit Question"):

    if "vector_db" not in st.session_state:
        st.warning("Load PDFs first.")

    elif not question:
        st.warning("Enter a question.")

    else:
        start = time.time()

        with st.spinner("Thinking..."):

            retriever = st.session_state["vector_db"].as_retriever(
                search_kwargs={"k": 8}
            )

            # -------- Query Rewrite (only when needed) --------
            if len(question.split()) < 6:
                query_prompt = f"""
Rewrite this question to be more precise for document search:
{question}
"""
                better_query = chat.invoke(query_prompt).content
            else:
                better_query = question

            # -------- Retrieval --------
            relevant_docs = retriever.invoke(better_query)

            # simple re-ranking
            relevant_docs = relevant_docs[:3]

            # -------- Context Builder --------
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"\nSOURCE {i+1}:\n{doc.page_content[:800]}\n"

            # -------- LLM Reasoning --------
            formatted_prompt = prompt.format(
                context=context,
                question=question
            )

            response = chat.invoke(formatted_prompt)

        end = time.time()

        st.subheader("Answer")
        st.write(response.content)

        st.info(f"Response time: {round(end-start,2)} seconds")
