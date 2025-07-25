from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag_core import load_and_split_docs, get_vectorstore, get_qa_chain, DATA_PATH, VECTORDB_PATH, OLLAMA_MODEL
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil

# --- FASTAPI SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest_files():
    docs = load_and_split_docs(DATA_PATH)
    vectordb = get_vectorstore(docs)
    vectordb.persist()
    return {"status": "Data ingested and indexed!"}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    vectordb = Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=OllamaEmbeddings(model=OLLAMA_MODEL)
    )
    qa = get_qa_chain(vectordb)
    answer = qa.run(req.question)
    return {"answer": answer}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(DATA_PATH, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}
