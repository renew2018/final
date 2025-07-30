import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 700

# Load model and DB
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading Groq LLM client...")
llm_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

print("Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Initialize FastAPI
app = FastAPI(title="NBC RAG Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Schema
class ChatRequest(BaseModel):
    collection_id: List[str]
    query: str
    top_k: int = 5


# Chat endpoint
@app.post("/chat")
def chat_with_nbc(req: ChatRequest):
    all_selected = []

    for coll_id in req.collection_id:
        try:
            collection = chroma_client.get_collection(name=coll_id)
        except Exception:
            continue  # Skip if collection not found

        query_embedding = embedder.encode(req.query).tolist()
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
        except Exception:
            continue  # Skip if query fails

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Match context
        matched = []
        for doc, meta in zip(documents, metadatas):
            if req.query.lower() in doc.lower() or "table 8" in doc.lower():
                matched.append((doc, meta))

        selected = matched if matched else list(zip(documents, metadatas))
        all_selected.extend(selected)

    if not all_selected:
        return {"answer": "No relevant context found in any selected collections."}

    # Build context string
    context_str = ""
    for i, (chunk, meta) in enumerate(all_selected):
        clause = meta.get("clause", "N/A")
        page = meta.get("page", "Unknown")
        context_str += f"[{i+1}] Page {page} | Clause {clause}:\n{chunk.strip()}\n\n"

    # Prompt
    prompt = f"""You are a senior building code consultant specializing in Indian and international building standards.

Your job is to answer user questions using only the provided context. You must ensure clarity, accuracy, and reference every answer to relevant clauses, pages, tables, and notes.

Strictly follow these rules for every response:

1. Use ONLY the context provided. Do not assume or fabricate any information.
2. Answer clearly and concisely in professional tone. Use bullet points or steps if necessary.
3. If applicable, include the exact clause number and page number.
4. If a figure or table is referenced, include:
   - Table or Figure number
   - Its title or a short summary
5. If any note (e.g., Note 6, Note 8) is mentioned in the context or in a table, explain the note clearly and include it under "Note Explanation".
6. If the answer is not available in the provided context, say:
   The provided context does not contain information relevant to this question.

Answer formatting must always follow this structure:

Clause: [Clause number]
Page: [Page number]
Answer:
[Clear explanation]
Note Explanation:
[If any note is referenced, include and explain it here]
Reference:
Clause title - Page number
Table/Figure (if any) - Title or summary

Important Style Rules:
- Do NOT use any of these: (), [], asterisks, dashes
- Do NOT use markdown
- Use only plain text
- Do NOT add personal opinions, small talk, or friendly phrases
- Keep it fact-based and direct
Context:
{context_str}

Question: {req.query}
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_TOKENS
        )
        return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")
