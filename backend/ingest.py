# ingest.py

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Initialization & Setup
# -----------------------------------------------------------------------------

# Load the embedding model globally at startup to avoid reloading it on every API call.
# The 384-dimensional all-MiniLM-L6-v2 model strikes a good balance between CPU performance 
# (~20ms latency) and accuracy for sentence-level semantic representations.
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

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

    Design Choice:
    pdfplumber was selected over PyPDF2/pdfminer because it handles multi-column 
    layouts and nested tables without breaking the reading order. This is critical 
    for preserving document context prior to chunking.

    Parameters:
        file_path (str): Path to the target PDF file.

    Returns:
        str: All extracted text separated by double newlines to mark page boundaries.
    """
    allText = []

    with pdfplumber.open(file_path) as currPDF:
        for currPage in currPDF.pages:
            # Fall back to an empty string if a page is unreadable (e.g., an un-scanned image)
            currPageText = currPage.extract_text() or ""
            
            if currPageText.strip():
                allText.append(currPageText.strip())
    
    # Double newlines act as an explicit visual boundary for the chunker
    return "\n\n".join(allText)


# -----------------------------------------------------------------------------
# Chunking Text
# -----------------------------------------------------------------------------

def chunkText(text: str, chunkSize: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a raw text string into a list of smaller, overlapping word blocks.

    Design Choice:
    Word-based chunking is chosen here as a straightforward, lightweight alternative 
    to token-based chunking. It avoids tokenization overhead while preventing sentences 
    from being split mid-word, which occurs with raw character splits. An overlap is 
    maintained to ensure context is preserved across boundary transitions.

    Parameters:
        text (str): The raw text to break down.
        chunkSize (int): Target word count per text slice. Default is 500.
        overlap (int): Number of words repeated between adjacent chunks. Default is 50.

    Returns:
        list[str]: A list containing the isolated text segments.
    """
    words = text.split()
    chunks = []
    step = chunkSize - overlap

    for i in range(0, len(words), step):
        chunkWords = words[i:i + chunkSize]
        chunks.append(" ".join(chunkWords))

        # Break early if the window reaches or overshoots the end of the text
        if i + chunkSize >= len(words):
            break
            
    return chunks


# -----------------------------------------------------------------------------
# Pipeline Core
# -----------------------------------------------------------------------------

def ingestPDF(filePath: str, docID: str) -> dict:
    """
    Executes the ingestion pipeline: reads a PDF, chunks the text, 
    generates vector embeddings, and registers the payload in ChromaDB.

    Design Choice:
    Metadata (docID and chunkIndex) is injected alongside the raw text. This provides 
    a querying mechanism to narrow searches to specific files later on, and offers a 
    clean hook for handling document deletions or overwrites.

    Parameters:
        filePath (str): Path to the uploaded document on disk.
        docID (str): Unique identifier for the document (typically the filename).

    Returns:
        dict: Ingestion summary containing metadata and processing status.
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

    print("[ingest] Generating embeddings...")
    # convert_to_list is omitted; calling .tolist() directly handles the NumPy conversion
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