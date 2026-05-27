# ingest.py

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Initialization & Setup
# -----------------------------------------------------------------------------

# Load the embedding model globally at startup to avoid reloading it on every API call.
# The 384-dimensional all-MiniLM-L6-v2 model strikes a good balance between CPU performance 
# and accuracy for sentence-level semantic representations.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Using a persistent client instead of an ephemeral one to ensure data survives 
# application restarts. Data is written directly to the local directory.
client = chromadb.PersistentClient(path="./chroma_db")

# Use a single collection for document tracking. get_or_create prevents duplication 
# errors when re-running the application.
collection = client.get_or_create_collection("documents")


# -----------------------------------------------------------------------------
# Extract Text
# -----------------------------------------------------------------------------

def extractTextFromPDF(file_path: str) -> str:
    """
    Extracts text from an on-disk PDF file and aggregates it into a single string.
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
    Executes the ingestion pipeline locally using the SentenceTransformer model weights.
    """
    print(f"[ingest] Extracting text from {filePath}...")
    text = extractTextFromPDF(filePath)
    
    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned document without a text layer."
        )
    
    print("[ingest] Chunking text...")
    chunks = chunkText(text)
    print(f"[ingest] Created {len(chunks)} chunks.")

    print("[ingest] Generating local embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

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