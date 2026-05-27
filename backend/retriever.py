# retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Shared Instances & Configuration
# -----------------------------------------------------------------------------

# Must use the exact same embedding model used during ingestion to ensure 
# vector comparisons occur within the identical geometric vector space.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Point to the existing database directory initialized by ingest.py
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")


# -----------------------------------------------------------------------------
# Core Retrieval
# -----------------------------------------------------------------------------

def retrieveRelevantChunks(query: str, docID: str, nResults: int = 4) -> list[dict]:
    """
    Embeds an incoming user query and fetches the top-N semantically similar 
    chunks restricted to a specific document.
    """
    # Vectorize the text query to match the shape of the stored document vectors
    queryVector = model.encode(query).tolist()

    # Query ChromaDB. We use metadata filtering (where=) to avoid cross-document 
    # leakage, and explicitly ask for "metadatas" to recover the chunk positions.
    results = collection.query(
        query_embeddings=[queryVector],
        n_results=nResults,
        where={"docID": docID},
        include=["documents", "distances", "metadatas"],
    )

    # Unpack the initial array wrapper
    chunksText = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print(f"[Retriever] Query: '{query[:60]}...")
    for i, (currChunk, currDist, currMeta) in enumerate(zip(chunksText, distances, metadatas)):
        preview = currChunk[:80].replace("\n", " ")
        print(f" [{i+1}] distance={currDist:.3f} | chunk={currMeta['chunkIndex']} | {preview}...")
    
    # Map the arrays back to structured dictionaries for consumers
    chunks = [
        {
            "text": currChunk,
            "chunkIndex": currMeta["chunkIndex"],
            "docID": currMeta["docID"],
        }
        for currChunk, currDist, currMeta in zip(chunksText, distances, metadatas)
    ]
    return chunks


# -----------------------------------------------------------------------------
# Management Utilities
# -----------------------------------------------------------------------------

def listIndexedDocuments() -> list[str]:
    """
    Scans the collection metadata to identify all unique documents indexed.
    """
    allItems = collection.get(include=["metadatas"])
    docIDs = sorted(set(meta["docID"] for meta in allItems["metadatas"]))
    return docIDs


def deleteDocument(docID: str) -> dict:
    """
    Purges all vector blocks associated with a specific document identifier.
    """
    indexedDoc = listIndexedDocuments()
    if docID not in indexedDoc:
        raise ValueError(f"Document '{docID}' not found in index.")
    
    collection.delete(where={"docID": docID})
    print(f"[Retriever] Deleted all chunks for doc '{docID}'.")
    
    return {"docID": docID, "status": "deleted"}