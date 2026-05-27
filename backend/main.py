# main.py

import json
import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ingest import ingestPDF
from retriever import retrieveRelevantChunks, listIndexedDocuments, deleteDocument
from llm import streamResponse

# -----------------------------------------------------------------------------
# App Initialization & Configuration
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Legal Document Analyzer",
    description="Upload a legal PDF and ask questions about it using RAG",
    version="2.0.0",
)

# Setup a local disk workspace for staging file streams prior to parsing.
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


# -----------------------------------------------------------------------------
# CORS Middleware Configuration
# -----------------------------------------------------------------------------

# Configured to permit localized cross-origin requests from the client tier.
# Middleware runs sequentially on all incoming requests and outgoing responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Restrict to the local development server origin
    allow_credentials=True,                   # Maintain support for session cookies/tokens if added
    allow_methods=["*"],                      # Expose all HTTP routing verbs (GET, POST, DELETE, etc.)
    allow_headers=["*"],                      # Allow arbitrary header injection (e.g., Auth, Content-Type)
)


# -----------------------------------------------------------------------------
# Data Models (Pydantic Validation schemas)
# -----------------------------------------------------------------------------

class MessageItem(BaseModel):
    """
    Represents a singular historical chat turn mapped directly to the OpenAI/Groq specification.
    """
    role: str       # "user" or "assistant"
    content: str    # Raw string text of the chat transmission


class AskRequest(BaseModel):
    """
    Validation schema for processing client queries against target vector indices.
    """
    query: str
    docID: str
    history: list[MessageItem] = []  # Defaults to an empty list to support stateless initial prompts


# -----------------------------------------------------------------------------
# REST Routing Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def healthCheck():
    """
    Basic service availability endpoint used by orchestrators to monitor system state.
    """
    return {"status": "ok", "message": "Legal Document Analyzer API v2 is running."}


@app.get("/documents")
def getDocuments():
    """
    Retrieves a listing of all active document references populated inside the vector store.
    """
    docIDs = listIndexedDocuments()
    return {"documents": docIDs}


@app.post("/upload")
async def uploadPDF(file: UploadFile = File(...)):
    """
    Receives multipart binary file data, caches it to disk, and runs the chunking-embedding pipeline.

    Design Choice:
    Uses 'async def' alongside streaming disk writes via 'shutil.copyfileobj' instead of reading 
    the whole file into RAM at once via an unconditional '.read()'. This optimization ensures that 
    the event loop remains non-blocking for simultaneous users when processing large file streams.
    """
    # Restrict input validation to document types with a textual layer
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF Files are accepted. Please upload a .pdf file!"
        )
    
    # Standardize directory path construction across divergent Operating Systems
    filePath = os.path.join(UPLOAD_DIRECTORY, file.filename)

    with open(filePath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        res = ingestPDF(filePath, docID=file.filename)
    except ValueError as e:
        # Gracefully handle validation failures raised downstream (e.g., scanned PDFs without text layers)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    
    return res


@app.post("/ask")
def askQuestion(request: AskRequest):
    """
    Accepts user prompts, retrieves localized text matches, and hooks into the LLM stream.

    Design Choice:
    Implements a two-phase data protocol over a singular HTTP response stream using an inner 
    generator wrapper. 
    - Phase 1: Immediately stringifies and yields a single line of raw metadata JSON.
    - Phase 2: Begins yielding raw completion tokens directly from the model stream.
    
    Sending data in this order allows the UI to parse the very first chunk for document references
    (enabling immediate citation rendering) without forcing the client to wait for text generation to finish.
    """
    # Enforce a fast-fail guard condition to confirm document presence before querying models
    allIndexedDocs = listIndexedDocuments()
    if request.docID not in allIndexedDocs:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.docID}' not found. Please upload it first.",
        )
    
    # Synchronously extract contextual references
    relevantChunks = retrieveRelevantChunks(
        query=request.query,
        docID=request.docID,
    )

    # Convert Pydantic abstractions into clean primitives required by the LLM client
    historyDicts = [message.model_dump() for message in request.history]

    def responseGenerator():
        # Step 1: Serialize references into a single newline-delimited stream boundary
        sources = [
            {"chunkIndex": c["chunkIndex"], "docID": c["docID"]} for c in relevantChunks
        ]
        yield json.dumps({"sources": sources}) + "\n"

        # Step 2: Unfurl text elements sequentially as they hit the connection pipe
        yield from streamResponse(request.query, relevantChunks, historyDicts)

    return StreamingResponse(
        responseGenerator(),
        media_type="text/plain"
    )


@app.delete("/documents/{docID}")
def deleteDoc(docID: str):
    """
    Wipes structural components and associated vectors tied to an isolated document namespace.

    Design Choice:
    Utilizes standard RESTful mapping convention (DELETE against a specific path parameter 
    resource) rather than routing operations through arbitrary utility POST paths. Path 
    variables are automatically decoded from URL spaces by the underlying router.
    """
    try:
        res = deleteDocument(docID)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return res