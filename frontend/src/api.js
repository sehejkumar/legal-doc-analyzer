// api.js

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// -----------------------------------------------------------------------------
// Document Retrieval
// -----------------------------------------------------------------------------

/**
 * Fetches an array of all active document identifiers indexed in the vector store.
 *
 * Design Choice:
 * Centralizing the base URL configuration via environment variables prevents hardcoding
 * errors. Vite handles these via `import.meta.env` and requires the `VITE_` prefix to
 * safely expose them to the client bundle.
 *
 * @returns {Promise<string[]>} Resolved list of document names.
 */
export async function fetchDocuments() {
  const response = await fetch(`${API_BASE}/documents`);

  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.statusText}`);
  }

  const data = await response.json();
  return data.documents;
}

// -----------------------------------------------------------------------------
// Document Ingestion
// -----------------------------------------------------------------------------

/**
 * Transmits a raw file object to the backend ingestion endpoint.
 *
 * Design Choice:
 * Multipart `FormData` is used here because binary data cannot be passed directly
 * within standard JSON payloads. We purposefully omit the `Content-Type` header
 * to let the browser configure its own dynamic multipart boundary flags automatically.
 *
 * @param {File} file - The browser file object selected from the DOM.
 * @returns {Promise<Object>} Ingestion summary payload from the backend.
 */
export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload Failed");
  }

  return response.json();
}

// -----------------------------------------------------------------------------
// Document Eviction
// -----------------------------------------------------------------------------

/**
 * Requests the permanent eviction of a document's vectors and components.
 *
 * Design Choice:
 * Filenames are forced through `encodeURIComponent` to protect the request path
 * string structure against breaks caused by special characters or spaces.
 *
 * @param {string} docID - Unique filename identifier.
 * @returns {Promise<Object>} Confirmation payload.
 */
export async function deleteDocument(docID) {
  const response = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(docID)}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Delete failed");
  }

  return response.json();
}

// -----------------------------------------------------------------------------
// Streaming Query Engine
// -----------------------------------------------------------------------------

/**
 * Establishes a text stream connection, decoding a dual-phase payload protocol.
 *
 * Design Choice:
 * The backend pipeline delivers its information sequentially: line 1 is a JSON metadata
 * payload string containing document reference objects, followed by continuous raw text deltas.
 * This service layer captures the stream via a unified `streamBuffer` string, extracts and
 * parses the metadata on the first newline boundary to run `onSources`, resets the frame,
 * and immediately routes all subsequent data fragments directly to the text UI loop via `onToken`.
 *
 * @param {string} query - The search prompt text.
 * @param {string} docID - Target document namespace filter.
 * @param {Array<Object>} history - Message history logs matching the backend model schema.
 * @param {Function} onSources - Triggered when the citation array is extracted.
 * @param {Function} onToken - Triggered as individual text string modifications land.
 * @param {Function} onDone - Stream closure callback.
 * @param {Function} onError - Pipeline failure fallback callback.
 */
export async function askQuestion(
  query,
  docID,
  history,
  onSources,
  onToken,
  onDone,
  onError,
) {
  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        docID: docID,
        history: history.map((m) => ({ role: m.role, content: m.content })),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Request Failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let metadataParsed = false;
    let streamBuffer = "";

    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        const finalLeftover = decoder.decode();
        if (finalLeftover && metadataParsed) {
          onToken(finalLeftover);
        }
        onDone();
        break;
      }

      // stream: true preserves internal multibyte character fragments across chunk boundaries
      streamBuffer += decoder.decode(value, { stream: true });

      if (!metadataParsed) {
        const newlineIndex = streamBuffer.indexOf("\n");

        if (newlineIndex !== -1) {
          const metadataLine = streamBuffer.slice(0, newlineIndex);
          const remainderText = streamBuffer.slice(newlineIndex + 1);

          try {
            const parsed = JSON.parse(metadataLine);
            onSources(parsed.sources || []);
          } catch {
            onSources([]);
          }

          metadataParsed = true;
          streamBuffer = "";

          if (remainderText) {
            onToken(remainderText);
          }
        }
      } else {
        if (streamBuffer) {
          onToken(streamBuffer);
          streamBuffer = "";
        }
      }
    }
  } catch (err) {
    onError(err.message);
  }
}
