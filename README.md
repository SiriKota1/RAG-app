# rag-app

Build a Retrieval-Augmented Generation (RAG) application using Ollama, LangChain, FastAPI, and Streamlit.

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI architecture that combines the power of large language models (LLMs) with external knowledge retrieval. Instead of relying solely on the model's internal knowledge, RAG retrieves relevant information from a custom knowledge base (such as your own documents) and feeds it to the LLM to generate more accurate, up-to-date, and context-aware responses.

**Key benefits:**
- Answers are grounded in your own data, not just the LLM's training set
- Easily update the knowledge base by adding new documents
- Reduces hallucinations and improves factual accuracy

## Features
- Uses Ollama for local LLM inference
- LangChain for orchestration and retrieval
- FastAPI for a production-ready backend API
- Streamlit for a simple web UI
- Easily ingest your own `.txt` files for retrieval

## Project Structure
```
rag-app/
├── app.py                # (legacy entrypoint, can be ignored)
├── src/
│   ├── app.py            # FastAPI backend
│   ├── streamlit_app.py  # Streamlit UI
│   └── rag_core.py       # Shared RAG logic
├── data/                 # Folder for your knowledge base (.txt files)
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup & Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Ollama** (make sure you have a model, e.g. llama3.2):
   ```bash
   ollama run llama3.2
   ```
3. **Add your knowledge:**
   - Place `.txt` files in the `data/` folder.

### Run the FastAPI backend
```bash
uvicorn src.app:app --reload
```
- **POST /ingest**: Ingest and index all `.txt` files in `data/`
- **POST /ask**: Ask a question (JSON: `{ "question": "..." }`)
- **POST /upload**: Upload a `.txt` file

### Run the Streamlit UI
```bash
streamlit run src/streamlit_app.py
```
- Use the sidebar to ingest data
- Ask questions in the main UI

---
