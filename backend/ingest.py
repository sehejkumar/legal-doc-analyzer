#ingest,py

'''
Purpose: take a pdf file, extract its text, split it into overlapping chunks,
convert each chunk into a vector embedding, and store everything in ChromaDB so 
it can be searched later.

This fule runs once per uploaded pdf ("indexing" phase). The "querying" phase
will search what we store here.
'''

import pdfplumber #extracts text from PDF files
import chromadb #Vector Database - stores and searches embeddings
from sentence_transformers import SentenceTransformer #converts text -> vectors

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load the embedding model
# ─────────────────────────────────────────────────────────────────────────────

'''
We load the model once at a module import time (when Python first reads this 
file), not inside a function. If we loaded it inside a function, it would 
re-download and re-initialize the model (~80MB neural net) on every single PDF
upload and would take 3-5 seconds each time. Loading once at startup time costs
only on first run, then it is in memory for all future calls.
'''

'''
What is "all-MiniLM-L6-v2":
 - A small but powerfful sentence embedding model.
  - MiniLM: compressed version of microsoft's MiniLM architecture
  - L6: 6 transformer layers
  - v2: version 2

It produces 384-dimensional vectors and runs on CPU in ~20ms per sentence.
First run downloads ~90MB from Hugging Face. Cached locally after that
'''

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Initialize ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

'''
ChromaDB has two modes:
- EphemeraClient() -> stores data only in RAM. Gone when program stops.
- PersistentClient() -> writes data to disk. Survives restarts

We will use PersistentClient() so if I have to restart the FASTAPI server, all
the previously indexed documents are still available.

path="./chroma_db" means ChromaDB will create a folder called "chroma_db" in the
backend directory and write its files there. This folder is what you would add
to gitignore (contains GB of vector data)
'''

client = chromadb.PersistentClient(path="./chroma_db")

'''
ChromaDB orgabizes data into "collections" (similar to tables in SQL). Each
collection holds a set of (text, vector, metadata) tuples. We use one
collection called "documents".

get_or_create_collection: 
- if "documents" already exists, use the existing one
- if it does not exist yet, creare a fresh one
this prevents errors on restart and avoids duplication of data
'''

collection = client.get_or_create_collection("documents")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extractTextFromPDF(file_path:str) -> str:
    '''
    Opens a PDF and returns all its text as a single string

    Why pdfplumber instead of PyPDF2 or pdfminer?
    pdfplumber is built on top of pdfminer but adds smarter text extractions
    It handles: multi-column layouts (common in legal contracts), tables(extracts
    them in reading order), preserved whitespace better than PyPDF2
    For legal documents with complex formatting, pdfplumber is the most reliable
    free option.

    PARAMETER:
      file_path: str  →  the path to the PDF on disk
                         (FastAPI will save uploaded files to a temp path)

    RETURNS:
      A single string with all text from all pages, separated by newlines.
    '''

    # We'll build up text page by page and join at the end.
    # Joining a list of strings with join() is much faster than using += in a loop.
    # (In Python, strings are immutable — every += creates a new string object.)
    allText = []

    # pdfplumber.open() is a context manager (the "with" keyword).
    # This ensures the file is properly closed even if an error occurs inside.
    # Same pattern as open() for regular files.

    with pdfplumber.open(file_path) as currPDF:
        # pdf.pages is a list of Page objects, one per page in the document.
        for currPage in currPDF.pages:
            # page.extract_text() reads all text from that page.
            # It returns None if the page has no extractable text (e.g. a scanned image).
            # The "or ''" handles that None case — if extract_text() returns None,
            # we use an empty string instead, avoiding a crash when we call .strip().
            currPageText = currPage.extract_text() or ""
            # .strip() removes leading/trailing whitespace and blank lines.
            # We only append if there's actual content (the "if text" check).
            if currPageText.strip():
                allText.append(currPageText.strip())
    
    # "\n\n" between pages creates a clear visual boundary in the combined text.
    # This matters for chunking — a double newline between pages prevents sentences
    # from accidentally merging across page boundaries.
    return "\n\n".join(allText)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunkText(text: str, chunkSize: int=500, overlap: int = 50) -> list[str]:
    """
    Splits a long string of text into overlapping chunks of approximately
    chunk_size words each.

    
    WHY WE CHUNK:
    1. Vector precision: a single embedding for an entire 40-page contract is useless.
       It averages everything into one undifferentiated blob. Individual chunks keep
       ideas focused and retrievable.
    2. LLM context limits: we can only pass so many tokens to the LLM at once.
       Retrieving 4 targeted chunks (~2000 words) is far more effective than
       trying to pass the whole document.

    WHY WORD-BASED, NOT CHARACTER OR TOKEN-BASED:
    Word boundaries are natural semantic units. Splitting by character count can
    cut words in half. Splitting by tokens requires running a tokenizer first
    (more complexity). Word-count is a good practical approximation.

    PARAMETERS:
      text       : str  →  the full document text
      chunk_size : int  →  target number of words per chunk (default 500)
      overlap    : int  →  number of words to repeat between adjacent chunks (default 50)

    RETURNS:
      A list of strings, each being one chunk of text.
    """

    # Split the entire text into individual words.
    # .split() with no argument splits on any whitespace (spaces, tabs, newlines)
    # and automatically ignores consecutive whitespace. Clean and simple.
    words = text.split()
    chunks=[]

    # We step through the word list using a sliding window.
    # Start at word 0, then jump forward by (chunk_size - overlap) each iteration.
    # The step is (chunk_size - overlap) NOT chunk_size, because we want consecutive
    # chunks to share `overlap` words.

    # Example with chunk_size=10, overlap=3:
    #   Chunk 1: words[0:10]   → words 0-9
    #   Chunk 2: words[7:17]   → words 7-16  (shares words 7,8,9 with chunk 1)
    #   Chunk 3: words[14:24]  → words 14-23 (shares words 14,15,16 with chunk 2)
    step= chunkSize - overlap

    for i in range(0, len(words), step):
        # Slice words from position i to i+chunk_size.
        # If i+chunk_size goes past the end of the list, Python just returns
        # whatever's left — no index error. This handles the last chunk cleanly.
        chunkWords = words[i:i+chunkSize]
        # Re-join the words back into a readable string.
        chunk = " ".join(chunkWords)
        chunks.append(chunk)

        # Stop condition: if we've reached (or passed) the end of the word list,
        # there's nothing more to chunk. Without this, the range() loop would
        # keep running with empty slices.
        if i + chunkSize >= len(words):
            break
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Embedding + Storage (the main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def ingestPDF (filePath: str, docID: str) -> dict:
    """
    Full pipeline: extract text → chunk → embed → store in ChromaDB.
    This is the function that FastAPI will call when a user uploads a PDF.

    PARAMETERS:
      filePath : str  →  path to the saved PDF on disk
      docID    : str  →  a unique identifier for this document
                          (we'll use the filename, e.g. "contract_2024.pdf")
                          Used to namespace chunks so we can delete or filter
                          by document later.
    RETURNS:
      A dict with metadata about what was ingested (useful for API responses).
    """

    #Step 1: Extract
    print(f"[ingest] Extracting text from {filePath}...")
    text = extractTextFromPDF(filePath)
    # Guard: if pdfplumber extracted nothing (scanned image PDF with no text layer),
    # raise a clear error rather than silently storing empty chunks.
    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned document without a text layer. "
            "OCR support is not yet implemented."
        )
    
    #Step 2: Chunk
    print("[ingest] Chunking text...")
    chunks = chunkText(text)
    print(f"[ingest] Created {len(chunks)} chunks.")

    #Step 3: Embed
    # model.encode() takes a list of strings and returns a numpy array of shape
    # (num_chunks, 384). Each row is the 384-dimensional embedding for one chunk.
    # show_progress_bar=True prints a tqdm progress bar to the terminal.
    # For a large document with 100+ chunks, this gives you visibility.
     # convert_to_list=True converts the numpy array to a plain Python list.
    # ChromaDB expects Python lists, not numpy arrays.
    print("[ingest] Generating embeddings...")
    embeddings = model.encode(chunks,show_progress_bar=True, convert_to_list=True)

    #Step 4: Build metadata
    # ChromaDB lets you attach arbitrary metadata to each stored item.
    # We store:
    #   - "doc_id"    : which document this chunk came from (for filtering)
    #   - "chunk_index": the position of this chunk within the document
    #                    (useful for ordering results or debugging)
    #
    # This metadata is not used for similarity search, but can be used to
    # filter results: "only search chunks from document X".

    metadatas=[{"docID": docID, "chunkIndex": i} for i,_ in enumerate(chunks)]

    #Step 5: Build IDs
    # Every item in ChromaDB needs a unique string ID.
    # We construct IDs as "doc_id__chunk_0", "doc_id__chunk_1", etc.
    # The double underscore (__) is a common separator convention — it's unlikely
    # to appear in a normal document name, reducing collision risk.
    #
    # These IDs serve two purposes:
    #   1. ChromaDB uses them as primary keys (no duplicates)
    #   2. We can use them to delete all chunks for a specific doc later:
    #      collection.delete(where={"doc_id": doc_id})

    ids = [f"{docID}__chunk_{i}" for i in range(len(chunks))]

    # ── Step 6: Store in ChromaDB ─────────────────────────────────────────────
    # collection.add() stores everything in one batch call.
    # ChromaDB writes to disk automatically (because we used PersistentClient).
    #
    # IMPORTANT: If the same doc_id is uploaded twice, these IDs will collide.
    # ChromaDB will raise an error. The production fix is to first delete existing
    # chunks for this doc_id before re-adding. For now, we keep it simple.
    print("[ingest] Storing in ChromaDB...")
    collection.add(ids=ids, documents=chunks,embeddings=embeddings,metadatas=metadatas,)
    print(f"[ingest] Done. Stored {len(chunks)} chunks for doc '{docID}'.")

    # Return a summary dict. FastAPI will serialize this to JSON and send it
    # back to the frontend as the response to the upload request.
    return {
        "docID": docID,
        "chunksStored": len(chunks),
        "status": "success",
    }