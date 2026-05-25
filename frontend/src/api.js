//api.js
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: All HTTP communication with the FastAPI backend lives here.
//          Components never call fetch() directly — they import from this file.
//
// WHY A SEPARATE SERVICE LAYER:
// If we scattered fetch() calls across components, changing the backend URL
// or adding auth headers would mean editing every component. Centralising API
// calls here means one change propagates everywhere. This is the "separation
// of concerns" principle — components handle UI, api.js handles transport.
//
// This pattern is standard in production React codebases and is worth
// mentioning in interviews: "I separated API calls into a service layer so
// components stay focused on rendering, not HTTP logic."
// ─────────────────────────────────────────────────────────────────────────────

// Base URL of the FastAPI backend.
// import.meta.env.VITE_API_URL reads from a .env file in the frontend folder.
// If not set, falls back to localhost:8000 for local development.
// In production you'd set VITE_API_URL=https://your-backend.railway.app
// in the deployment platform's environment variables.
//
// WHY import.meta.env INSTEAD OF process.env:
// Vite uses ES modules, not CommonJS (Node). process.env is a Node concept.
// Vite exposes env variables via import.meta.env. Variables must be prefixed
// with VITE_ to be exposed to the browser (for security — you don't want
// server-only secrets leaking into frontend bundles).

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// fetchDocuments — GET /documents
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetches the list of all document IDs currently indexed in ChromaDB.
 * Called on app load to populate the document selector.
 *
 * RETURNS: Promise<string[]>  e.g. ["contract.pdf", "nda.pdf"]
 */

export async function fetchDocuments() {
  const response = await fetch(`${API_BASE}/documents`);
  // response.ok is true for 2xx status codes (200, 201, etc.).
  // For any other status (404, 500, etc.), we throw an error with the
  // status text so the caller gets a meaningful message rather than
  // silently getting undefined data.
  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.statusText}`);
  }
  // response.json() parses the JSON body into a JavaScript object.
  // The backend returns: { "documents": ["contract.pdf", ...] }
  // We destructure to get just the array.
  const data = await response.json();
  return data.documents;
}

// ─────────────────────────────────────────────────────────────────────────────
// uploadPdf — POST /upload
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Uploads a PDF file to the backend for ingestion.
 * The backend runs the full RAG pipeline: extract → chunk → embed → store.
 *
 * PARAMETERS:
 *   file : File  →  a browser File object from an <input type="file"> element
 *
 * RETURNS: Promise<{doc_id: string, chunks_stored: number, status: string}>
 *
 * WHY FormData INSTEAD OF JSON:
 * File uploads can't be sent as JSON — JSON is text-only and can't represent
 * binary data efficiently. FormData is the browser's way of sending multipart
 * form data, which can include both text fields and binary file data in one
 * request. FastAPI's UploadFile expects exactly this format.
 */

export async function uploadPDF(file) {
  // FormData() creates a multipart form data object.
  // .append("file", file) adds the File object under the key "file".
  // "file" must match the parameter name in FastAPI: file: UploadFile = File(...)
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    // NOTE: Do NOT set Content-Type header manually for FormData.
    // The browser sets it automatically including the boundary string:
    // Content-Type: multipart/form-data; boundary=----WebKitFormBoundary...
    // If you set it manually, you'll break the boundary and FastAPI won't
    // be able to parse the file. This is a very common beginner mistake.
    body: formData,
  });

  if (!response.ok) {
    // Parse the error detail from FastAPI's error response format:
    // { "detail": "Only PDF files are accepted..." }
    const error = await response.json();
    throw new Error(error.detail || "Upload Failed");
  }
  return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// askQuestion — POST /ask (streaming)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sends a question to the backend and streams the response token by token.
 * This is the most complex function in the frontend — it handles the
 * streaming ReadableStream from FastAPI's StreamingResponse.
 *
 * PARAMETERS:
 *   query    : string    →  the user's question
 *   docId    : string    →  which document to query against
 *   onToken  : function  →  callback called with each text token as it arrives
 *                           e.g. (token) => setAnswer(prev => prev + token)
 *   onDone   : function  →  callback called when the stream is complete
 *   onError  : function  →  callback called if an error occurs
 *
 * WHY CALLBACKS INSTEAD OF RETURNING A VALUE:
 * We can't return the full response because it doesn't exist yet —
 * it's arriving token by token over several seconds. Callbacks let the
 * Chat component update state incrementally as each token arrives,
 * which is what creates the real-time typewriter effect.
 *
 * This pattern (passing callbacks for async incremental updates) is also
 * used in WebSockets and EventSource-based streaming.
 */

export async function askQuestion(query, docID, onToken, onDone, onError) {
  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      // For JSON bodies we DO set Content-Type manually.
      // This tells FastAPI to parse the body as JSON (not form data).
      headers: { "Content-Type": "application/json" },
      // JSON.stringify converts the JS object to a JSON string.
      // Must match the Pydantic model: { query: string, doc_id: string }
      // Note: JavaScript uses camelCase (docId) but the API uses snake_case (doc_id).
      // We translate here so components can use JS conventions.
      body: JSON.stringify({ query, docID: docID }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Request Failed");
    }

    // ── Read the streaming response ─────────────────────────────────────────
    // response.body is a ReadableStream — a browser API for reading data
    // incrementally as it arrives, rather than waiting for the full response.
    //
    // .getReader() returns a ReadableStreamDefaultReader.
    // We call reader.read() in a loop — each call returns a Promise that
    // resolves to { value: Uint8Array, done: boolean } when the next chunk
    // arrives. done=true means the stream has ended.
    const reader = response.body.getReader();
    // TextDecoder converts raw bytes (Uint8Array) to a JavaScript string.
    // "utf-8" is the encoding — matches what FastAPI sends.
    // We create it once outside the loop for efficiency.
    const decoder = new TextDecoder("utf-8");

    //The reading loop. Runs until the stream is exhausted (done=true).
    while (true) {
      // reader.read() is async — it waits for the next chunk to arrive
      // from the server before resolving. This is where the function "pauses"
      // while the LLM is generating the next token.
      const { value, done } = await reader.read();
      //done=true means the server closed the stream (generation finished)
      if (done) {
        onDone();
        break;
      }

      // value is a Uint8Array (raw bytes). Decode to string.
      // stream=true means "this might be part of a larger string, keep
      // the internal buffer if the chunk ends mid-character". Important for
      // multi-byte UTF-8 characters (e.g. em-dashes in legal documents).
      const token = decoder.decode(value, { stream: true });

      //Call the onToken callback with the decoded text.
      //The Chat component uses this to append to the message being built
      onToken(token);
    }
  } catch (err) {
    onError(err.message);
  }
}
