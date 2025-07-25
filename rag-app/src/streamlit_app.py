import streamlit as st
from rag_core import load_and_split_docs, get_vectorstore, get_qa_chain, DATA_PATH, VECTORDB_PATH, OLLAMA_MODEL
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil

st.title("🦙 RAG App with Ollama, LangChain, Streamlit & FastAPI")

st.sidebar.header("Data Ingestion")

uploaded_file = st.sidebar.file_uploader("Upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

if st.sidebar.button("Ingest Data"):
    with st.spinner("Loading and indexing documents..."):
        docs = load_and_split_docs(DATA_PATH)
        vectordb = get_vectorstore(docs)
        vectordb.persist()
    st.sidebar.success("Data ingested and indexed!")

st.header("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving answer..."):
        vectordb = Chroma(
            persist_directory=VECTORDB_PATH,
            embedding_function=OllamaEmbeddings(model=OLLAMA_MODEL)
        )
        qa = get_qa_chain(vectordb)
        answer = qa.run(question)
    st.success(answer)

st.info(
    "To add knowledge, upload .txt, .pdf, or .docx files in the sidebar and click 'Ingest Data'."
)
