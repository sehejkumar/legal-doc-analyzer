// Upload.jsx

import { useState, useRef } from "react";
import { uploadPDF } from "../api";

// -----------------------------------------------------------------------------
// Document Upload & Ingestion Dropzone Component
// -----------------------------------------------------------------------------

/**
 * Handles file staging, drag-and-drop validation thresholds, file validation,
 * and interface state switching during the background vector indexing pipeline.
 *
 * Design Choice:
 * File type verification runs immediately at the browser event level (checking MIME type 
 * against 'application/pdf') for both manual input selection and window drops. This fast-fail 
 * workflow prevents large, unparsable payloads from hitting network bandwidth streams.
 *
 * @param {Function} onUploadSuccess - Callback passing the indexed document ID back up to the root layout.
 * @returns {JSX.Element} Landing landing layout containing dropzone controls and upload tracking.
 */
export default function Upload({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    if (file.type !== "application/pdf") {
      setStatus("error");
      setErrorMsg("Only PDF files are supported.");
      return;
    }
    
    setSelectedFile(file);
    setStatus("idle");
    setErrorMsg("");
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (!file) return;
    
    if (file.type !== "application/pdf") {
      setStatus("error");
      setErrorMsg("Only PDF files are supported.");
      return;
    }
    
    setSelectedFile(file);
    setStatus("idle");
    setErrorMsg("");
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setStatus("uploading");
    setErrorMsg("");
    
    try {
      const result = await uploadPDF(selectedFile);
      setStatus("success");
      // Slight delay execution to allow completion visual states to register nicely on the UI
      setTimeout(() => onUploadSuccess(result.docID), 900);
    } catch (err) {
      setStatus("error");
      setErrorMsg(err.message);
    }
  }

  return (
    <div className="upload-page">
      <div className="upload-content">

        <div className="upload-header">
          <div className="upload-eyebrow">Legal Document Intelligence</div>
          <h1 className="upload-title">Ask your contract<br />anything.</h1>
          <p className="upload-subtitle">
            Upload a legal document — contract, NDA, lease, terms of service —
            and query it in plain language. Answers are grounded exclusively
            in your document.
          </p>
        </div>

        <div
          className={`upload-zone ${dragOver ? "upload-zone--dragover" : ""} ${selectedFile ? "upload-zone--selected" : ""}`}
          onClick={() => fileInputRef.current.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          {selectedFile ? (
            <div className="upload-file-info">
              <div className="upload-file-icon">⚖</div>
              <div className="upload-file-details">
                <div className="upload-file-name">{selectedFile.name}</div>
                <div className="upload-file-size">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB · PDF
                </div>
              </div>
              <div className="upload-file-check">✓</div>
            </div>
          ) : (
            <div className="upload-zone-prompt">
              <div className="upload-zone-icon">↑</div>
              <div className="upload-zone-text">Drop a PDF here</div>
              <div className="upload-zone-sub">or click to browse</div>
            </div>
          )}
        </div>

        {status === "error" && (
          <div className="upload-error">{errorMsg}</div>
        )}

        <button
          className={`upload-cta ${status}`}
          onClick={handleUpload}
          disabled={!selectedFile || status === "uploading" || status === "success"}
        >
          {status === "idle" && "Analyze Document"}
          {status === "uploading" && (
            <span className="upload-cta-loading">
              <span className="upload-cta-dot" />
              Indexing document…
            </span>
          )}
          {status === "success" && "✓ Opening…"}
          {status === "error" && "Try Again"}
        </button>

        {status === "uploading" && (
          <p className="upload-processing-note">
            Extracting text, generating embeddings, storing in vector index.
            Large documents may take 20–40 seconds.
          </p>
        )}

        <div className="upload-features">
          <span className="upload-feature-pill">RAG-grounded answers</span>
          <span className="upload-feature-pill">Semantic search</span>
          <span className="upload-feature-pill">Source citations</span>
          <span className="upload-feature-pill">Runs locally</span>
        </div>

      </div>
    </div>
  );
}