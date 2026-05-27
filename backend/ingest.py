# ingest.py

import os
import pdfplumber
import chromadb
from groq import Groq

# -----------------------------------------------------------------------------
# Initialization & Setup
# -----------------------------------------------------------------------------

# Initialize the Groq cloud client. This offloads the embedding calculations
# to Groq's hardware, keeping our local Render container's RAM usage ultra-low.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Using a persistent client instead of an ephemeral one to ensure data survives 
# application restarts. Data is written directly to the local directory.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a single collection for document tracking. get_or_create prevents duplication 
# errors when re-running the application.
collection = chroma_client.get_or_create_collection("documents")


# -----------------------------------------------------------------------------
# Extract Text
# -----------------------------------------------------------------------------

def extractTextFromPDF(file_path: str) -> str:
    """
    Extracts text from an on-disk PDF file and aggregates it into a single string.

    Design Choice:
    pdfplumber handles multi-column layouts and nested tables without breaking 
    the reading order. This is critical for preserving document context prior to chunking.
    """
    allText = []

    with pdfplumber.open(file_path) as currPDF:
        for currPage in currPDF.pages:
            currPageText = currPage.extract_text() or ""
            
            if currPageText.strip():
                allText.append(currPageText.strip())
    
    return "\n\n".join(allText)


# -----------------------------------------------------------------------------
# Chunking Text
# -----------------------------------------------------------------------------

def chunkText(text: str, chunkSize: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a raw text string into a list of smaller, overlapping word blocks.
    An overlap is maintained to ensure context is preserved across transitions.
    """
    words = text.split()
    chunks = []
    step = chunkSize - overlap

    for i in range(0, len(words), step):
        chunkWords = words[i:i + chunkSize]
        chunks.append(" ".join(chunkWords))

        if i + chunkSize >= len(words):
            break
            
    return chunks


# -----------------------------------------------------------------------------
# Pipeline Core
# -----------------------------------------------------------------------------

def ingestPDF(filePath: str, docID: str) -> dict:
    """
    Executes the ingestion pipeline: reads a PDF, chunks the text, 
    generates cloud vector embeddings via Groq, and registers the payload in ChromaDB.
    """
    print(f"[ingest] Extracting text from {filePath}...")
    text = extractTextFromPDF(filePath)
    
    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned document without a text layer. "
            "OCR support is not yet implemented."
        )
    
    print("[ingest] Chunking text...")
    chunks = chunkText(text)
    print(f"[ingest] Created {len(chunks)} chunks.")

    print("[ingest] Generating embeddings via Groq Cloud API...")
    embeddings = []
    
    # Send each text chunk to Groq's high-speed embedding API.
    for chunk in chunks:
        # Replacing newlines minimizes formatting layout noise inside the text embedding vectors
        cleaned_text = chunk.replace("\n", " ")
        
        response = groq_client.embeddings.create(
            model="nomic-embed-text-v1.5",
            input=cleaned_text
        )
        # Extract the array of vector floats
        embeddings.append(response.data[0].embedding)

    # Create isolation filtering scopes and tracking indexes
    metadatas = [{"docID": docID, "chunkIndex": i} for i, _ in enumerate(chunks)]

    # Generate isolated primary keys using a double-underscore convention to avoid name collisions
    ids = [f"{docID}__chunk_{i}" for i in range(len(chunks))]

    print("[ingest] Storing in ChromaDB...")
    collection.add(
        ids=ids, 
        documents=chunks, 
        embeddings=embeddings, 
        metadatas=metadatas
    )
    print(f"[ingest] Done. Stored {len(chunks)} chunks for doc '{docID}'.")

    return {
        "docID": docID,
        "chunksStored": len(chunks),
        "status": "success",
    }