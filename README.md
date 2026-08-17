# RAG Document Chatbot

> **Status:** Core system complete and functional locally. Public deployment in progress.

### Overview
A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from a document set. The system embeds documents into a FAISS vector store, retrieves the most relevant chunks for each question, and generates grounded answers with source citations — never hallucinating outside the provided documents.

### Business Value
Lets any business turn its PDFs, FAQs, policies, or product docs into a chatbot that gives customers instant, accurate answers with sources — reducing support workload without risking made-up answers.

### Tech Stack
* **FastAPI + Uvicorn** — REST API backend (`/ask`, `/health` endpoints)
* **FAISS** — vector similarity search over embedded document chunks
* **SentenceTransformers** (`all-MiniLM-L6-v2`) — lightweight local embeddings, no API cost
* **Groq (Llama 3.1 8B)** — fast LLM inference for answer generation
* **Vanilla JS/HTML frontend** — clean chat interface

### Architecture
1. **Ingestion:** documents are chunked and embedded offline; the FAISS index (`faiss_index.index`) and chunk store (`chunk_texts.pkl`) are shipped with the app for fast cold starts.
2. **Retrieval:** each query is embedded and matched against the index (top-k = 3 chunks).
3. **Generation:** a strict system prompt forces the model to answer only from retrieved context — if the answer isn't in the documents, it says so explicitly.
4. **API:** `POST /ask` returns the answer plus the source chunks used, so responses are auditable.

### Structure
```
backend/    FastAPI app, FAISS index, chunk store, deployment config
frontend/   Chat UI (single-page)
```
