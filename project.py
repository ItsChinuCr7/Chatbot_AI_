import streamlit as st
import time
import os

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# =============================
# CONFIG
# =============================
PDF_PATH = "D:/Chatbot_AI/test"
INDEX_PATH = "faiss_index"


# =============================
# LOAD OLLAMA MODEL
# =============================
chat = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434"
)


# =============================
# EMBEDDING MODEL
# =============================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
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
st.title("📄 Advanced Local RAG (Ollama + FAISS)")
question = st.text_input("Ask a question")


# =============================
# MEMORY (Conversation Context)
# =============================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# =============================
# DOCUMENT TRANSFORMATION
# =============================
def transform_documents(docs):
    cleaned_docs = []
    for doc in docs:
        text = doc.page_content
        text = text.replace("\n\n", " ")
        text = " ".join(text.split())
        doc.page_content = text
        cleaned_docs.append(doc)
    return cleaned_docs


# =============================
# SYNTHETIC QA GENERATION
# =============================
def generate_qa_pairs(docs):

    qa_docs = []

    for doc in docs[:15]:   # safety limit

        text = doc.page_content[:1000]

        q_prompt = f"""
Generate 3 different questions from the following text:

{text}

Return only questions.
"""

        raw_questions = chat.invoke(q_prompt).content.split("\n")

        questions = [
            q.strip("- ").strip()
            for q in raw_questions
            if len(q.strip()) > 5 and "?" in q
        ]

        for q in questions:

            a_prompt = f"""
Answer the question using ONLY this text.

Text:
{text}

Question:
{q}
"""

            answer = chat.invoke(a_prompt).content

            qa_docs.append(
                Document(
                    page_content=f"Question: {q}\nAnswer: {answer}",
                    metadata=doc.metadata
                )
            )

    return qa_docs


# =============================
# LOAD / BUILD VECTOR DATABASE
# =============================
@st.cache_resource
def load_vector_db():

    # load existing index
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(
        path=PDF_PATH,
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

    # ---- Synthetic QA generation ----
    qa_docs = generate_qa_pairs(docs)

    docs.extend(qa_docs)

    vector_db = FAISS.from_documents(docs, embeddings)

    vector_db.save_local(INDEX_PATH)

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

            # ---- Query rewrite only if short ----
            if len(question.split()) < 6:
                rewrite_prompt = f"""
Rewrite this question to be more precise for document search:
{question}
"""
                better_query = chat.invoke(rewrite_prompt).content
            else:
                better_query = question

            relevant_docs = retriever.invoke(better_query)
            relevant_docs = relevant_docs[:3]

            # ---- Build context ----
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"\nSOURCE {i+1}:\n{doc.page_content[:800]}\n"

            history_text = "\n".join(
                st.session_state["chat_history"][-5:]
            )

            formatted_prompt = prompt.format(
                context=context + "\n" + history_text,
                question=question
            )

            response = chat.invoke(formatted_prompt)

            # save limited memory
            st.session_state["chat_history"].append(
                f"User: {question}\nAssistant: {response.content}"
            )
            st.session_state["chat_history"] = \
                st.session_state["chat_history"][-5:]

        end = time.time()

        st.subheader("Answer")
        st.write(response.content)
        st.info(f"Response time: {round(end-start,2)} seconds")
