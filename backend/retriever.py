# retriever.py

import os
import chromadb
from groq import Groq

# -----------------------------------------------------------------------------
# Shared Instances & Configuration
# -----------------------------------------------------------------------------

# Initialize the Groq cloud client. Must use the exact same embedding model
# used during ingestion to ensure geometric vector calculations align correctly.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Point to the existing database directory initialized by ingest.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")


# -----------------------------------------------------------------------------
# Core Retrieval
# -----------------------------------------------------------------------------

def retrieveRelevantChunks(query: str, docID: str, nResults: int = 4) -> list[dict]:
    """
    Embeds an incoming user query via Groq cloud API and fetches the top-N 
    semantically similar chunks restricted to a specific document.
    """
    # Vectorize the text query string using the cloud model
    cleaned_query = query.replace("\n", " ")
    response = groq_client.embeddings.create(
        model="nomic-embed-text-v1.5",
        input=cleaned_query
    )
    queryVector = response.data[0].embedding

    # Query ChromaDB. We use metadata filtering (where=) to avoid cross-document 
    # leakage, and explicitly ask for "metadatas" to recover the chunk positions.
    results = collection.query(
        query_embeddings=[queryVector],
        n_results=nResults,
        where={"docID": docID},
        include=["documents", "distances", "metadatas"],
    )

    # Unpack the initial array wrapper (ChromaDB structures responses for batch inputs)
    chunksText = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Console logging for tracking semantic match quality during development.
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
    Queries only the metadata field to optimize memory allocation performance.
    """
    allItems = collection.get(include=["metadatas"])
    
    # Extract unique identifiers using a set lookup
    docIDs = sorted(set(
        meta["docID"]
        for meta in allItems["metadatas"]
    ))

    return docIDs


def deleteDocument(docID: str) -> dict:
    """
    Purges all vector blocks associated with a specific document identifier.
    """
    indexedDoc = listIndexedDocuments()
    if docID not in indexedDoc:
        raise ValueError(f"Document '{docID}' not found in index.")
    
    # Target and wipe elements bound to the requested metadata scope
    collection.delete(where={"docID": docID})
    print(f"[Retriever] Deleted all chunks for doc '{docID}'.")
    
    return {"docID": docID, "status": "deleted"}