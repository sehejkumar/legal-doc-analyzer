#retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Given a user's question, find the most semantically relevant chunks
#          from the document that was previously indexed by ingest.py.
#
# This is the "Retrieval" step in Retrieval-Augmented Generation.
# It runs on EVERY user message, before the LLM is called.
#
# Data flow:
#   user question (string)
#       → embed question → query vector (384 numbers)
#       → ChromaDB cosine similarity search
#       → top-k most relevant chunks (strings)
#       → returned to llm.py which builds the prompt
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Shared instances
# ─────────────────────────────────────────────────────────────────────────────

# Both ingest.py and retriever.py use the same model name. Python's import
# system caches modules so the model is only loaded into RAM once.
# CRITICAL: must be the same model as ingest.py — vectors are only comparable
# within the same model's geometric space.

model = SentenceTransformer("all-MiniLM-L6-v2")

# Connects to the existing chroma_db folder that ingest.py already populated.
# We are NOT creating a new database — just getting a handle to query it.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: The retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def retrieveRelevantChunks(query: str, docID: str, nResults: int=4) -> list[str]:
    '''
    Embeds a user query and returns the n most semantically similar text
    chunks from a specific document.

    PARAMETERS:
        query    : str  →  the user's question in plain English
        docID    : str  →  which document to search within
        nResults : int  →  how many chunks to return (default 4)

    RETURNS:
        A list of strings — the raw text of the most relevant chunks.
        These will be injected into the LLM prompt.
    '''
    #Step 1: Embed the query
    # Convert the question into the same 384-dim vector space the chunks live in.
    # .tolist() converts numpy array → plain Python list (what ChromaDB expects).
    queryVector = model.encode(query).tolist()

    #Step 2: Query ChromaDB
    # query_embeddings expects List[List[float]] — double-nested — because
    # ChromaDB supports batch queries. Single query still needs outer list: [vec].

    # where={"doc_id": doc_id} is metadata filtering — scopes the search to
    # only chunks from this specific document. Without it, all indexed documents
    # would be searched and results would bleed across documents.

    # include= limits what gets returned. We skip embeddings (wasteful to
    # return 384 floats per chunk when we only need the text).

    results = collection.query(
        query_embeddings=[queryVector],
        n_results=nResults,
        where={"docID": docID},
        include=["documents", "distances"],
    )

    #Step 3: Unpack results
    # results["documents"] is List[List[str]] — double nested because ChromaDB
    # supports batch queries. [0] gets the results for our single query.

    # distances are cosine DISTANCE (not similarity): distance = 1 - similarity
    # Lower = better match. Results are already sorted best-first.

    chunks = results["documents"][0] #List[str]
    distances = results["distances"][0] #List[float]

    #Step 4: Debug Logging
    # Print distances during dev to verify retrieval quality.
    # < 0.3 = strong match, 0.3-0.6 = moderate, > 0.7 = poor retrieval.
    # Most RAG failures are retrieval failures — monitor this closely.

    print(f"[Retriever] Query: '{query[:60]}...")
    for i, (currChunk, currDist) in enumerate(zip(chunks, distances)):
        preview = currChunk[:80].replace("\n", " ")
        print(f" [{i+1}] distance={currDist:.3f} | {preview}...")
    
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Utility — list all indexed documents
# ─────────────────────────────────────────────────────────────────────────────

def listIndexedDocuments() -> list[str]:
    '''
    Returns a list of unique doc_ids currently stored in ChromaDB.
    Used by the frontend to populate a document selector dropdown.
    Queries metadata only — does not fetch embeddings or chunk text.
    '''

    allItems = collection.get(include=["metadatas"])
    # all_items["metadatas"] is a list of dicts:
    # [{"docID": "contract.pdf", "chunkIndex": 0}, ...]
    # set() deduplicates, sorted() gives consistent ordering.

    docIDs = sorted(set(
        meta["docID"]
        for meta in allItems["metadatas"]
    ))

    return docIDs