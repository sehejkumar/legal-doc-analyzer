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

def retrieveRelevantChunks(query: str, docID: str, nResults: int=4) -> list[dict]:
    """
    Embeds a user query and returns the n most semantically similar chunks
    from a specific document — now including source metadata.

    RETURN FORMAT CHANGE (v1 → v2):
    v1 returned:  ["chunk text 1", "chunk text 2", ...]
    v2 returns:   [
                    { "text": "chunk text 1", "chunk_index": 4, "doc_id": "contract.pdf" },
                    { "text": "chunk text 2", "chunk_index": 11, "doc_id": "contract.pdf" },
                    ...
                    ]

    WHY RETURN DICTS INSTEAD OF PLAIN STRINGS:
    The source highlighting feature needs to show the user WHERE in the document
    each answer came from. chunk_index tells us which numbered chunk it was.
    By bundling text + metadata together in one object, callers don't need to
    manage two parallel lists — everything travels together.
    This is the "keep related data together" principle from data modelling.

    PARAMETERS:
        query     : str  →  the user's question
        doc_id    : str  →  which document to search within
        n_results : int  →  how many chunks to return (default 4)

    RETURNS:
        List[dict] with keys: text, chunk_index, doc_id
    """
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

    # CHANGE: added "metadatas" to include= so we get chunk_index back
    # Previously: include=["documents", "distances"]
    # Now:        include=["documents", "distances", "metadatas"]
    # Metadatas contains the dict we stored during ingest:
    # {"doc_id": "contract.pdf", "chunk_index": 4}

    results = collection.query(
        query_embeddings=[queryVector],
        n_results=nResults,
        where={"docID": docID},
        include=["documents", "distances", "metadatas"],
    )

    #Step 3: Unpack results
    # results["documents"] is List[List[str]] — double nested because ChromaDB
    # supports batch queries. [0] gets the results for our single query.

    # distances are cosine DISTANCE (not similarity): distance = 1 - similarity
    # Lower = better match. Results are already sorted best-first.

    chunksText = results["documents"][0] #List[str]
    distances = results["distances"][0] #List[float]
    metadatas = results["metadatas"][0] #List[dict]

    #Step 4: Debug Logging
    # Print distances during dev to verify retrieval quality.
    # < 0.3 = strong match, 0.3-0.6 = moderate, > 0.7 = poor retrieval.
    # Most RAG failures are retrieval failures — monitor this closely.

    print(f"[Retriever] Query: '{query[:60]}...")
    for i, (currChunk, currDist, currMeta) in enumerate(zip(chunksText, distances, metadatas)):
        preview = currChunk[:80].replace("\n", " ")
        print(f" [{i+1}] distance={currDist:.3f} | chunk={currMeta['chunkIndex']} | {preview}...")
    
    # Zip the three parallel lists into a list of dicts.
    # zip() takes multiple iterables and yields tuples: (text, dist, meta)
    # We only include text and metadata in the output — distance is for
    # internal logging only, not exposed to the LLM or frontend.

    chunks = [
        {
            "text": currChunk,
            "chunkIndex": currMeta["chunkIndex"],
            "docID": currMeta["docID"],
        }
        for currChunk, currDist, currMeta in zip(chunksText, distances, metadatas)
    ]
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

def deleteDocument(docID: str) -> dict:
    '''
    Deletes ALL chunks belonging to a specific document from ChromaDB.
    Called by the DELETE /documents/{doc_id} endpoint in main.py.

    WHY THIS IS SAFE:
    ChromaDB's .delete(where=...) uses the same metadata filter as .query().
    It deletes only chunks where doc_id matches — other documents are untouched.

    WHAT HAPPENS TO THE DATA:
    ChromaDB removes the vectors, text, and metadata from its on-disk store.
    The chroma_db/ folder shrinks. The document can be re-uploaded and
    re-indexed from scratch at any time.

    RETURNS:
        {"doc_id": doc_id, "status": "deleted"}
    '''

    # First check the document exists — better to fail with a clear message
    # than silently delete nothing and return success.
    indexedDoc = listIndexedDocuments()
    if docID not in indexedDoc:
        raise ValueError(f"Document '{docID}' not found in index.")
    
    # collection.delete(where=...) removes all items matching the filter.
    # This is the same where= syntax as collection.query() — consistent API.
    collection.delete(where={"docID": docID})
    print(f"[Retriever] De;eted all chunks for doc '{docID}'.")
    return {"docID": docID, "status": "deleted"}