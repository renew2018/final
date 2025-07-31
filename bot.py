import os
from typing import List
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
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 700

# Load Models and DB
embedder = SentenceTransformer(EMBED_MODEL)
llm_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

# Initialize FastAPI
app = FastAPI(title="NBC RAG Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# === Chat API ===
class ChatRequest(BaseModel):
    collection_id: List[str]
    query: str
    top_k: int = 5

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

        def is_partial_match(query, text):
            return any(q in text.lower() for q in query.lower().split())

        def fuzzy_match(query, text, threshold=0.6):
            return SequenceMatcher(None, query.lower(), text.lower()).ratio() > threshold

        matched = []
        for doc, meta in zip(documents, metadatas):
            if is_partial_match(req.query, doc) or fuzzy_match(req.query, doc):
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
Additional Prompt Handling Rule:
When the user input is a statement or unclear, use basic common sense to rephrase it into a proper, grammatically correct question.
Apply basic language understanding even if the user's input lacks proper punctuation or grammar. Focus on identifying the key parameters and reframe the input into a clear, grammatically correct question.

If the query includes technical terms, search for the most contextually relevant clause, even if the clause number or related table isn't directly mentioned nearby. Always prioritize contextual accuracy over superficial keyword matches.
Strictly follow these rules for every response:
1. If the user query is incomplete, fragmented, or written like a keyword phrase (e.g., “30 mtrs height mercantile building pressurization of staircase is required”), reframe it into a full, grammatically correct question before answering.
2. Always display the reframed question at the top under "Reframed Question:".
3. Use ONLY the context provided. Do not assume or fabricate any information.
4. Answer clearly and concisely in professional tone. Use bullet points or steps if necessary.
5. If applicable, include the exact clause number and page number.
6. If a figure or table is referenced, include:
   - Table or Figure number
   - Its title or a short summary
7. If any note (e.g., Note 6, Note 8) is mentioned in the context or in a table, explain the note clearly and include it under "Note Explanation".
   - If no note is mentioned in the context or in a table. Do not explain any note. Do not show the Note Explanation section in the answer.
8. Apply basic common sense when answering. Do not include irrelevant details, unnecessary repetition, or overly complicated language. Only include what is needed to clearly and accurately answer the question.
9. If the user query contains a keyword or phrase that partially matches a clause or table entry (e.g., "gym", "swimming", "weight room"), identify the full matching entry (e.g., "Gym, stadium (play area), Health club, weight rooms") and reframe the query using that.
10. Always extract and return all values (e.g., Ez, Rp, Ra, Notes) from the full matched entry, not the partial keyword alone. Treat it as a structured lookup.
11.If the answer is not available in the provided context, say:
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

# === PDF Library APIs ===
PDF_DIR = "Standards"

@app.get("/api/list-pdfs")
async def list_pdfs():
    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    return {"pdfs": pdfs}

@app.get("/api/list-collections")
async def list_collections():
    try:
        collections = chroma_client.list_collections()
        collection_names = [coll.name for coll in collections]
        return {"collections": collection_names}
    except Exception as e:
        return {"error": str(e)}
@app.get("/pdfs/{filename}")
async def serve_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    return {"error": "File not found"}
