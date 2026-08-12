"""
RAG Chatbot Backend API
Free-tier FastAPI app for Project 4 portfolio.
"""
import os
import pickle
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─── Config ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is required")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunk_texts.pkl"
TOP_K = 3

# ─── Load models & index at startup ─────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunk texts...")
with open(CHUNKS_PATH, "rb") as f:
    chunk_texts = pickle.load(f)

print(f"Ready. Index has {index.ntotal} vectors.")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a document assistant. Answer the user's question strictly using the provided context below.
If the answer is not contained in the context, respond exactly with: \"I cannot find this in the documents.\"
Do not use any outside knowledge or make assumptions beyond what is stated."""

# ─── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation backend for Project 4",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str
    k: int = 3

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

def retrieve(query: str, k: int = TOP_K):
    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idxs = index.search(qvec, k)
    return [chunk_texts[i] for i in idxs[0]]

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content

@app.get("/")
def root():
    return {"status": "RAG Backend is running", "vectors": index.ntotal}

@app.get("/health")
def health():
    return {"status": "ok", "index_size": index.ntotal}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        chunks = retrieve(req.query, k=req.k)
        answer = generate_answer(req.query, chunks)
        return AskResponse(answer=answer, sources=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
