import os
import io
import re
import uuid
import json
import fitz  
import pdfplumber
import pytesseract
import nltk
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === Setup ===
nltk.download("punkt")
load_dotenv()
CHROMA_DIR = "chroma_store"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Models ===
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
client = PersistentClient(path=CHROMA_DIR)

# === FastAPI App ===
app = FastAPI(
    title="Enhanced PDF Extractor",
    description="PDF Clause and Table Extractor",
    version="3.0"
)

# === Utility Functions ===
def clean_paragraphs(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip() and not re.search(r"Supply Bureau.*valid upto", line)]

def extract_clause_blocks(text):
    clause_pattern = re.compile(r'(\d{1,2}(?:\.\d+)+(\([a-z]\))?)\s+([A-Z][^\n]{5,})')
    matches = list(clause_pattern.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_number = match.group(1)
        clause_title = match.group(3).strip()
        paragraphs = clean_paragraphs(text[start:end])
        blocks.append({
            "clause_number": clause_number,
            "clause_title": clause_title,
            "paragraphs": paragraphs
        })
    return blocks

def extract_tables_from_page(pdf_path: str, page_number: int, page_text: str) -> List[Dict[str, Any]]:
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                raw_tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines"
                })

                table_titles = re.findall(r'(Table\s*\d+[A-Z\-]*)(?:\s*[:\-\u2013]\s*(.*))?', page_text, re.IGNORECASE)

                for idx, tbl in enumerate(raw_tables):
                    if not tbl or len(tbl) < 2:
                        continue

                    header, *rows = tbl
                    clean_columns = [c.strip() if c else "" for c in header]
                    clean_rows = [[c.strip() if c else "" for c in row] for row in rows]

                    if idx < len(table_titles):
                        table_number = table_titles[idx][0].strip()
                        table_title = table_titles[idx][1].strip() if table_titles[idx][1] else ""
                        full_title = f"{table_number}: {table_title}" if table_title else table_number
                    else:
                        table_number = f"Table {idx+1}"
                        table_title = ""
                        full_title = f"{table_number} on page {page_number + 1}"

                    tables.append({
                        "title": full_title,
                        "table_number": table_number,
                        "table_title": table_title,
                        "columns": clean_columns,
                        "rows": clean_rows,
                        "notes": [],
                        "extracted_from": "pdfplumber"
                    })
    except Exception as e:
        print(f"[pdfplumber error] Page {page_number}: {e}")
    return tables

def extract_figures(text):
    figures = re.findall(r'(Fig(?:ure)?\.?\s*\d+[^:\n]*)', text, re.IGNORECASE)
    result = []
    for fig in figures:
        parts = fig.split(None, 2)
        if len(parts) >= 3:
            result.append({"figure_number": parts[1].strip("."), "title": parts[2].strip()})
        elif len(parts) == 2:
            result.append({"figure_number": parts[1].strip("."), "title": ""})
    return result

def process_pdf(pdf_path: str) -> List[dict]:
    doc = fitz.open(pdf_path)
    structured_data = []
    for page_num in tqdm(range(len(doc)), desc="Processing PDF Pages"):
        page = doc.load_page(page_num)
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        if not text or len(text.strip()) < 20:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

        text_cleaned = "\n".join(clean_paragraphs(text))
        clause_blocks = extract_clause_blocks(text_cleaned)
        tables = extract_tables_from_page(pdf_path, page_num, text_cleaned)
        figures = extract_figures(text_cleaned)

        if clause_blocks:
            for block in clause_blocks:
                block.update({"page": page_num + 1, "tables": tables, "figures": figures})
                structured_data.append(block)
        else:
            structured_data.append({
                "clause_number": "",
                "clause_title": "",
                "page": page_num + 1,
                "paragraphs": clean_paragraphs(text_cleaned),
                "tables": tables,
                "figures": figures
            })
    return structured_data

def embed_and_store(data: List[dict], collection_name: str) -> int:
    collection = client.get_or_create_collection(name=collection_name)
    count = 0
    for doc in tqdm(data, desc="Embedding and Storing"):
        parts = []
        if doc.get("clause_number"):
            parts.append(f"Clause {doc['clause_number']}: {doc.get('clause_title', '')}")
        parts += doc.get("paragraphs", [])

        for table in doc.get("tables", []):
            parts.append(f"Table {table.get('table_number', '')}: {table.get('table_title', '')}")
            if "columns" in table:
                parts.append(" | ".join(table["columns"]))
            parts += [" | ".join(row) for row in table.get("rows", [])]

        for fig in doc.get("figures", []):
            parts.append(f"Figure {fig.get('figure_number')}: {fig.get('title', '')}")

        full_text = " ".join(parts).strip()
        if not full_text:
            continue

        embedding = model.encode(full_text).tolist()
        metadata = {
            "page": doc.get("page", 0),
            "clause": doc.get("clause_number", ""),
            "title": doc.get("clause_title", "")
        }

        collection.add(
            documents=[full_text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())]
        )
        count += 1
    return count

# === Endpoints ===

@app.get("/collections", summary="List ChromaDB collections")
def list_collections():
    return [c.name for c in client.list_collections()]

@app.post("/upload_pdf", summary="Upload PDF and embed to ChromaDB")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    if not collection_name.strip():
        raise HTTPException(status_code=400, detail="Collection name cannot be empty.")

    filename = file.filename
    temp_path = f"temp_{uuid.uuid4()}.pdf"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extract structured data (text, clauses, tables, etc.)
        structured_data = process_pdf(temp_path)

        # Save JSON
        json_filename = f"{Path(filename).stem}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        # Embed into ChromaDB
        count = embed_and_store(structured_data, collection_name)

        return JSONResponse(
            content={
                "message": f"Processed '{filename}' and embedded {count} entries to '{collection_name}'",
                "collection": collection_name,
                "count": count,
                "output_json": json_filename
            },
            media_type="application/json"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
@app.post("/embed_pages", summary="Embed specific pages into ChromaDB")
async def embed_multiple_pages(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    page_numbers: str = Form(...),  # Example: "311,312,314"
):
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        structured_data = []
        pages_to_process = [int(p.strip()) for p in page_numbers.split(",") if p.strip().isdigit()]

        with fitz.open(temp_path) as doc:
            for page_number in pages_to_process:
                if page_number < 1 or page_number > len(doc):
                    continue

                page = doc.load_page(page_number - 1)
                try:
                    text = page.get_text("text")
                except Exception:
                    text = ""

                if not text or len(text.strip()) < 20:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img)

                text_cleaned = "\n".join(clean_paragraphs(text))
                clause_blocks = extract_clause_blocks(text_cleaned)
                tables = extract_tables_from_page(temp_path, page_number - 1, text_cleaned)
                figures = extract_figures(text_cleaned)

                if clause_blocks:
                    for block in clause_blocks:
                        block.update({
                            "page": page_number,
                            "tables": tables,
                            "figures": figures
                        })
                        structured_data.append(block)
                else:
                    structured_data.append({
                        "clause_number": "",
                        "clause_title": "",
                        "page": page_number,
                        "paragraphs": clean_paragraphs(text_cleaned),
                        "tables": tables,
                        "figures": figures
                    })

        count = embed_and_store(structured_data, collection_name)

        return JSONResponse({
            "message": f"Re-embedded {len(pages_to_process)} pages into collection '{collection_name}'",
            "collection": collection_name,
            "pages": pages_to_process,
            "count": count
        })

    finally:
        try:
            os.remove(temp_path)
        except PermissionError:
            print(f"[Warning] Could not delete {temp_path} — file still in use.")

@app.delete("/delete_collection/{collection_name}", summary="Delete ChromaDB collection")
def delete_chroma_collection(collection_name: str):
    try:
        client.delete_collection(name=collection_name)
        return {"message": f"Deleted collection '{collection_name}'"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to delete collection: {str(e)}")
