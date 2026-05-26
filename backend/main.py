# main.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: The FastAPI application. This is the HTTP layer that wires together
#          ingest.py, retriever.py, and llm.py and exposes them as API endpoints
#          that the React frontend can call.
#
# ENDPOINTS:
#   GET  /                          → health check
#   GET  /documents                 → list all indexed document IDs
#   POST /upload                    → upload + index a PDF
#   POST /ask                       → ask a question about a document (streaming)
#
# Run with:
#   uvicorn main:app --reload
#   (--reload restarts the server automatically when you save any .py file)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CHANGES FROM V1:
#   - AskRequest now includes history: list[dict] for conversation memory
#   - /ask endpoint passes history + chunk metadata to llm.stream_response()
#   - /ask now streams source metadata FIRST, then the text response
#   - New endpoint: DELETE /documents/{doc_id}
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: App initialization
# ─────────────────────────────────────────────────────────────────────────────

# FastAPI() creates the application instance.
# title, description, and version appear in the auto-generated API docs at
# http://localhost:8000/docs (Swagger UI) and /redoc (ReDoc).
# These docs are generated for free from your code — no extra work needed.
# You can test every endpoint directly from the browser there.

app = FastAPI(
    title="Legal Document Analyzer",
    description="Upload a legal PDF and ask questions about it using RAG",
    version="2.0.0",
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CORS middleware
# ─────────────────────────────────────────────────────────────────────────────

# WHAT CORS IS:
# Browsers enforce the Same-Origin Policy: JavaScript can only make fetch()
# calls to the same origin (protocol + host + port) the page was served from.
# Your frontend is at localhost:5173 and backend at localhost:8000 — different
# ports = different origins = browser blocks the request by default.
#
# CORSMiddleware adds the Access-Control-Allow-Origin header to every response,
# telling the browser "I explicitly permit requests from these origins."
#
# WHAT "middleware" MEANS:
# Middleware is code that runs on EVERY request/response, before it hits your
# route handler and after your handler returns. Think of it as a pipeline:
#
#   browser request
#       → CORS middleware (adds headers, handles preflight OPTIONS requests)
#       → your route function
#       → CORS middleware (adds headers to response)
#       → browser receives response
#
# Other common middleware: authentication, logging, rate limiting, compression.

app.add_middleware(
    CORSMiddleware,
    # allow_origins: which frontend URLs are permitted to call this API.
    # In development we allow localhost:5173 (Vite's default port).
    # In production you'd change this to your actual frontend domain:
    # ["https://yourdomain.com"]
    # NEVER use ["*"] (allow all) in production for an API with real data —
    # it means any website on the internet could call your backend.
    allow_origins=["http://localhost:5173"],
    # allow_credentials=True: permits cookies and Authorization headers to be
    # sent cross-origin. Needed if you add user authentication later.
    allow_credentials=True,
    # allow_methods: which HTTP methods are permitted.
    # ["*"] means GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD — all of them.
    # You could restrict to ["GET", "POST"] for tighter security.
    allow_methods=["*"],
    # allow_headers: which HTTP request headers are permitted cross-origin.
    # ["*"] allows all headers including Content-Type, Authorization, etc.
    allow_headers=["*"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Upload directory setup
# ─────────────────────────────────────────────────────────────────────────────

# We need somewhere to temporarily save uploaded PDF files so pdfplumber can
# read them from disk. FastAPI receives files as bytes in memory — pdfplumber
# needs a file path.
#
# UPLOAD_DIR is defined as a constant so it's easy to change.
# os.makedirs(..., exist_ok=True) creates the directory if it doesn't exist.
# exist_ok=True means "don't raise an error if it already exists" — safe to
# call on every startup.
#
# Note: "uploads/" is in .gitignore so PDFs never get committed to git.
# In production you'd use cloud storage (S3, GCS) instead of local disk.

UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Request/Response models (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

# WHAT PYDANTIC IS:
# Pydantic is a data validation library built into FastAPI. You define the
# shape of request bodies as Python classes that inherit from BaseModel.
# FastAPI automatically:
#   1. Parses the incoming JSON into your class
#   2. Validates that all required fields are present and the right type
#   3. Returns a clear 422 Unprocessable Entity error if validation fails
#   4. Documents the expected schema in the Swagger UI
#
# This replaces manually doing: data = request.json(); query = data["query"]
# with automatic parsing, validation, and documentation.

class MessageItem(BaseModel):
    """
    Represents one message in the conversation history.
    Matches the {role, content} shape the Groq API expects.
 
    WHY A SEPARATE MODEL:
    Pydantic validates nested objects too. By defining MessageItem separately,
    FastAPI validates that every item in the history list has exactly these
    two fields with the correct types — not just that history is a list.
    """
    role: str   # "user" or "assistant"
    content: str    # the message text

class AskRequest(BaseModel):
    '''
    Request body for POST /ask.
 
    FIELDS:
      query  : str  →  the user's question in plain English
      docID  : str  →  which document to search (must match a previously
                        uploaded document's filename)
    '''
    """
    CHANGES FROM V1:
    Added history field — the full prior conversation turns.
 
    history is Optional with a default of [] (empty list).
    This means:
      - First message in a conversation: history=[] or omit history entirely
      - Subsequent messages: history=[all prior {role,content} turns]
      - Backwards compatible: old clients not sending history still work
    """
    query: str
    docID: str
    history: list[MessageItem] = []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Routes (endpoints)
# ─────────────────────────────────────────────────────────────────────────────

# WHAT A ROUTE IS:
# A route maps an HTTP method + URL path to a Python function.
# The decorator @app.get("/") means: when the server receives a GET request
# to the path "/", call the function below it and return its result as JSON.
#
# FastAPI automatically serializes Python dicts/lists/Pydantic models to JSON.
# You just return Python objects — FastAPI handles the rest.

#Route 1: Health Check

@app.get("/")
def healthCheck():
    '''
    Simple health check endpoint.

    WHY THIS EXISTS:
    Every production API has a health check endpoint. Deployment platforms
    (Railway, Render, AWS) ping it periodically to verify the server is alive.
    If it returns a non-200 status, the platform restarts the server.
    It's also the first thing you hit in your browser to verify the server started.

    Try it: http://localhost:8000/
    '''
    return {"status": "ok", "message": "Legal Document Analyzer API v2 is running."}

#Route 2: List indexed documents

@app.get("/documents")
def getDocuments():
    '''
    Returns a list of all document IDs currently indexed in ChromaDB.
 
    The frontend calls this on load to populate the document selector dropdown.
    If no documents have been uploaded yet, returns an empty list.
 
    Try it: http://localhost:8000/documents
    '''
    docIDs = listIndexedDocuments()
    return {"documents": docIDs}

#Route 3: Upload and index a PDF

@app.post("/upload")
async def uploadPDF(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, saves it to disk, runs the full ingestion
    pipeline (extract → chunk → embed → store), and returns a summary.

    WHAT UploadFile IS:
    FastAPI's UploadFile wraps the uploaded file with metadata:
    file.filename  → original filename ("contract.pdf")
    file.content_type → MIME type ("application/pdf")
    await file.read() → the raw bytes of the file

    WHAT File(...) MEANS:
    File() tells FastAPI this parameter comes from multipart form data
    (not JSON body). The ... (Ellipsis) means the field is required.
    If the request doesn't include a file, FastAPI returns 422 automatically.

    WHY async/await:
    File I/O is slow relative to CPU operations. async def lets FastAPI handle
    other requests while waiting for the file to be saved to disk — instead of
    blocking the entire server. await file.read() and the shutil copy both
    involve waiting for disk operations.

    In a single-user dev environment this doesn't matter much. In production
    with hundreds of concurrent users, async is what keeps the server responsive.

    RETURNS:
    {"docID": "contract.pdf", "chunksStored": 47, "status": "success"}
    """

    # ── Validate file type ────────────────────────────────────────────────────
    # Only accept PDFs. content_type is set by the browser based on the file
    # extension. This is a basic check — a more robust check would inspect the
    # file's magic bytes (the first 4 bytes of a PDF are always "%PDF").
    #
    # HTTPException is FastAPI's way of returning an error response.
    # Raising it immediately stops the function and sends the error to the client.
    # status_code=400 means "Bad Request" — the client sent invalid input.

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF Files are accepted. Please upload a .pdf file!"
        )
    
    # ── Save file to disk ─────────────────────────────────────────────────────
    # We need the file on disk because pdfplumber.open() takes a file path.
    # file.filename gives us the original name the user uploaded (e.g. "contract.pdf").
    # We use it as both the save path and the docID for ChromaDB.
    #
    # os.path.join() builds the full path: "uploads/contract.pdf"
    # Always use os.path.join() instead of string concatenation for paths —
    # it handles OS differences (Windows uses \, Unix uses /).

    filePath = os.path.join(UPLOAD_DIRECTORY, file.filename)

    # shutil.copyfileobj() copies the file bytes from the upload stream to disk.
    # We open the destination file in "wb" (write binary) mode.
    # file.file is the raw SpooledTemporaryFile object FastAPI wraps.
    # This is more memory-efficient than await file.read() for large files
    # because it streams chunks rather than loading everything into RAM at once.

    with open(filePath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ── Run the ingestion pipeline ────────────────────────────────────────────
    # ingestPDF() from ingest.py does: extract text → chunk → embed → store.
    # We wrap it in try/except to catch errors (e.g. scanned PDFs with no text)
    # and return a clean error message rather than a 500 server crash.
    #
    # The docID is just the filename. Simple and human-readable.
    # In a multi-user production system you'd use a UUID to avoid collisions
    # between users uploading files with the same name.

    try:
        res = ingestPDF(filePath,docID=file.filename)
    except ValueError as e:
        #ingestPDF raises ValueError for PDFs with no extractable text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=599, detail=f"Ingestion failed: {str(e)}")
    
    return res

# ── Route 4: Ask a question (streaming) ──────────────────────────────────────

@app.post("/ask")
def askQuestion(request: AskRequest):
    """
    Takes a user question and doc_id, retrieves relevant chunks from ChromaDB,
    calls the Groq LLM, and streams the response back token by token.

    WHY THIS IS NOT async:
    StreamingResponse handles its own async streaming internally. The generator
    function stream_response() is synchronous (it uses a regular for loop over
    the Groq stream). Marking this route async would require making the generator
    async too (using "async for" and "async def"), which adds complexity without
    benefit here. Sync routes work fine for streaming.

    WHAT StreamingResponse DOES:
    Instead of collecting the full response and returning it at once,
    StreamingResponse keeps the HTTP connection open and writes each yielded
    token to the response body immediately. The browser reads this as a
    streaming response using the Fetch API's ReadableStream interface.

    media_type="text/plain":
    Tells the browser the content is plain text being streamed, not JSON.
    The frontend will read it character by character and append to the chat UI.

    RETURNS:
    An HTTP 200 response with a streaming body of plain text tokens.
    """

    """
    CHANGES FROM V1:
    1. Passes request.history to stream_response() for conversation memory.
    2. Streams source metadata as the FIRST chunk before text tokens.
       This lets the frontend know which chunks were used immediately,
       display source chips before the answer starts streaming in.
 
    HOW THE STREAMING PROTOCOL WORKS NOW:
    The response stream has two phases:
 
    Phase 1 — metadata (one JSON line):
      {"sources": [{"chunk_index": 4, "doc_id": "contract.pdf"}, ...]}
 
    Phase 2 — text tokens (many small strings):
      "According" " to" " Excerpt" " 2" "," " the" " termination" ...
 
    The frontend reads the first line as JSON to get sources, then treats
    everything after as streaming text to append to the message.
 
    WHY SEND METADATA FIRST (not last):
    If we sent metadata after the text, the frontend would need to buffer
    the entire response before knowing the sources — defeating streaming.
    Sending it first means source chips appear immediately, then text flows in.
 
    WHY USE A GENERATOR WRAPPER:
    StreamingResponse needs a single generator. We create a wrapper generator
    that first yields the metadata JSON line, then yields all text tokens
    from stream_response(). One clean stream, two logical phases.
    """

    # ── Validate the doc_id exists ────────────────────────────────────────────
    # Before running expensive embedding and LLM calls, verify the requested
    # document actually exists in ChromaDB. If the user passes a doc_id that
    # was never indexed, ChromaDB would return zero results and the LLM would
    # say "I could not find information" — confusing. Better to fail fast with
    # a clear error.

    allIndexedDocs = listIndexedDocuments()
    if request.docID not in allIndexedDocs:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.docID}' not found. Please upload it first.",
        )
    
    # ── Retrieve relevant chunks ──────────────────────────────────────────────
    # retrieve_relevant_chunks() from retriever.py:
    #   1. Embeds the query
    #   2. Searches ChromaDB for top-4 most similar chunks
    #   3. Returns them as a list of strings
    # This happens synchronously before streaming starts.

    relevantChunks = retrieveRelevantChunks(
        query=request.query,
        docID=request.docID,
    )

    # Convert Pydantic MessageItem objects to plain dicts for llm.py
    # Pydantic models need .model_dump() to become plain Python dicts.
    # llm.py expects List[dict], not List[MessageItem].
    historyDicts = [message.model_dump() for message in request.history]

    def responseGenerator():
        """
        A generator that yields:
          1. One JSON line with source metadata
          2. All streaming text tokens from the LLM
 
        The \n at the end of the metadata line is the delimiter.
        The frontend splits on the first \n to separate metadata from text.
        """
        # Phase 1: send sources as JSON
        # Extract just chunk_index and doc_id — that's all the frontend needs
        sources = [
            {"chunkIndex": c["chunkIndex"], "docID": c["docID"]} for c in relevantChunks
        ]

        # json.dumps converts the Python dict to a JSON string.
        # \n is the delimiter — the frontend reads until the first newline
        # to get the metadata, then treats everything after as streaming text.
        yield json.dumps({"sources": sources}) + "\n"
        #Phase 2: stream LLm response tokens
        yield from streamResponse(request.query, relevantChunks,historyDicts)

    # ── Stream the LLM response ───────────────────────────────────────────────
    # stream_response() from llm.py is a generator that:
    #   1. Builds the grounded prompt (system + context + user question)
    #   2. Calls the Groq API with stream=True
    #   3. Yields text tokens one by one as they arrive
    #
    # StreamingResponse wraps that generator and handles forwarding each token
    # to the browser over the open HTTP connection.
    #
    # The frontend will receive this as a stream it can read with:
    #   const reader = response.body.getReader()
    #   while (true) { const { value, done } = await reader.read(); ... }

    return StreamingResponse(
        responseGenerator(),
        media_type="text/plain"
    )

#Route 5: Delete all chunks for a document
@app.delete("/document/{docID}")
def deleteDoc(docID: str):
    """
    Deletes all indexed chunks for a document from ChromaDB.
 
    WHY DELETE VERB AND PATH PARAMETER:
    REST convention: the resource is /documents/{doc_id}.
    DELETE method on a resource URL = delete that resource.
    Path parameters ({doc_id}) are used for resource identifiers in REST.
    Query parameters (?doc_id=...) are for filtering, not identification.
 
    This is a RESTful design decision worth explaining in interviews:
    "I used DELETE /documents/{doc_id} rather than POST /delete because
    REST maps HTTP verbs to CRUD operations — DELETE is semantically correct
    and makes the API self-documenting."
 
    IMPORTANT — URL ENCODING:
    doc_id values are filenames like "my contract.pdf". Spaces and special
    characters must be URL-encoded by the client: "my%20contract.pdf".
    FastAPI automatically URL-decodes path parameters before passing them
    to this function, so we receive the original filename.
    """
    try:
        res = deleteDocument(docID)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return res