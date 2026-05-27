// Sidebar.jsx

import { useState } from "react";
import { deleteDocument } from "../api";

// -----------------------------------------------------------------------------
// Sidebar Navigation Panel Component
// -----------------------------------------------------------------------------

/**
 * Lists indexed documents, manages deletion lifecycles with stateful confirmation 
 * gates, and exposes navigation handlers to switch between documents or trigger uploads.
 *
 * Design Choice:
 * String truncation isolates the file extension during name slicing. This guarantees 
 * that critical formatting information (e.g., '.pdf') remains readable on the UI 
 * layout even when a long filename is condensed.
 *
 * @param {Array<string>} documents - Collection of document names parsed from database storage.
 * @param {string|null} activeDocId - The currently selected document key namespace.
 * @param {Function} onSelect - Updates root state tracking when a document button is chosen.
 * @param {Function} onDelete - Notifies root elements to clean up chat states on successful deletion.
 * @param {Function} onNewUpload - Forces root container layout redirection back to the upload lane.
 * @returns {JSX.Element} Interactive contextual aside sidebar control block.
 */
export default function Sidebar({ documents, activeDocId, onSelect, onDelete, onNewUpload }) {
  const [confirmingDelete, setConfirmingDelete] = useState(null);
  const [deleting, setDeleting] = useState(null);

  async function handleDelete(docId) {
    setDeleting(docId);
    try {
      await deleteDocument(docId);
      onDelete(docId);
    } catch (err) {
      console.error("Delete failed:", err.message);
    } finally {
      setDeleting(null);
      setConfirmingDelete(null);
    }
  }

  function truncate(name, max = 24) {
    if (name.length <= max) return name;
    const ext = name.lastIndexOf(".");
    const base = ext > 0 ? name.slice(0, ext) : name;
    const extension = ext > 0 ? name.slice(ext) : "";
    return base.slice(0, max - 3 - extension.length) + "…" + extension;
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-icon">⚖</span>
        <span className="sidebar-brand-name">LexAI</span>
      </div>

      <div className="sidebar-section-label">Documents</div>

      <nav className="sidebar-nav">
        {documents.map(docId => (
          <div
            key={docId}
            className={`sidebar-item ${activeDocId === docId ? "sidebar-item--active" : ""}`}
          >
            <button
              className="sidebar-item-btn"
              onClick={() => onSelect(docId)}
              title={docId}
            >
              <span className="sidebar-item-icon">📄</span>
              <span className="sidebar-item-name">{truncate(docId)}</span>
            </button>

            {confirmingDelete === docId ? (
              <div className="sidebar-delete-confirm">
                <button
                  className="sidebar-confirm-yes"
                  onClick={() => handleDelete(docId)}
                  disabled={deleting === docId}
                >
                  {deleting === docId ? "…" : "Remove"}
                </button>
                <button
                  className="sidebar-confirm-no"
                  onClick={() => setConfirmingDelete(null)}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                className="sidebar-delete-btn"
                onClick={() => setConfirmingDelete(docId)}
                title="Remove document"
              >
                ×
              </button>
            )}
          </div>
        ))}
      </nav>

      <button className="sidebar-upload-btn" onClick={onNewUpload}>
        <span>+</span> Upload Document
      </button>
    </aside>
  );
}