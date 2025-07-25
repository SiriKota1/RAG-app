from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os
import shutil

OLLAMA_MODEL = "llama3.2"
DATA_PATH = "./data"
VECTORDB_PATH = "./chroma_db"

def load_and_split_docs(data_path: str):
    docs = []
    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path, fname)
        if fname.endswith(".txt"):
            print(f"Loading TXT: {fname}")
            loader = TextLoader(fpath)
            docs.extend(loader.load())
        elif fname.endswith(".pdf"):
            print(f"Loading PDF: {fname}")
            loader = PyPDFLoader(fpath)
            docs.extend(loader.load())
        elif fname.endswith(".docx"):
            print(f"Loading DOCX: {fname}")
            loader = Docx2txtLoader(fpath)
            docs.extend(loader.load())
    print(f"Total documents loaded: {len(docs)}")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(split_docs)}")
    return split_docs

def get_vectorstore(docs):
    # Remove old index before creating a new one
    if os.path.exists(VECTORDB_PATH):
        shutil.rmtree(VECTORDB_PATH)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectordb = Chroma.from_documents(
        docs, embeddings, persist_directory=VECTORDB_PATH
    )
    return vectordb

def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
