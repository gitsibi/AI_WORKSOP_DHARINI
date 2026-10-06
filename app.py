# Needed on Streamlit Cloud (Linux) because Chroma requires a newer sqlite3.
# Skipped automatically on Windows, where pysqlite3 is not installed.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).parent / "turing_machine.pdf"
MODEL_NAME = "gemini-3.8-flash"  # change here if Google retires this name

st.set_page_config(page_title="Turing Machine Chatbot", page_icon="🤖")
st.title("Turing Machine RAG Chatbot")
st.caption("Ask questions about the Turing Machines document.")


@st.cache_resource
def build_retriever():
    # 1. LOAD the PDF
    pages = PyPDFLoader(str(PDF_PATH)).load()

    # 2. SPLIT into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # 3. EMBED and STORE in Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="turing_chatbot",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource
def build_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.1,
        google_api_key=api_key,
    )


def extract_text(content):
    """Gemini 3.x can return a list of blocks; keep only the text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def ask_question(question: str, retriever, llm):
    # RETRIEVE relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # AUGMENT the prompt with the context
    prompt = f"""
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say:
"I don't have that information in the document."

Context from document:
{context}

Question:
{question}

Answer:
"""
    # GENERATE the answer
    response = llm.invoke(prompt)
    return extract_text(response.content), docs


# ---------- Setup checks ----------
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "GOOGLE_API_KEY is not configured. Add it in .streamlit/secrets.toml "
        "locally, or under App settings > Secrets on Streamlit Cloud."
    )
    st.stop()

if not PDF_PATH.exists():
    st.error(f"Could not find the document: {PDF_PATH.name}")
    st.stop()

retriever = build_retriever()
llm = build_llm(api_key)

# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.write(f"- {s}")

question = st.chat_input("Ask a question about Turing Machines...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                answer, documents = ask_question(question, retriever, llm)
        except Exception as e:
            st.error(f"Could not get an answer: {e}")
            st.stop()

        st.markdown(answer)
        sources = [d.page_content[:150].replace("\n", " ") for d in documents]
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )