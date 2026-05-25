// App.jsx
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Root component. Manages which view is shown (Upload or Chat)
//          and passes data between them.
//
// APP STATE MACHINE:
//   "upload"  →  user sees the PDF upload screen
//   "chat"    →  user sees the chat interface for the uploaded document
//
// This is a simple two-state machine. The transition is:
//   upload → chat   : triggered by onUploadSuccess(docId)
//   chat → upload   : triggered by onReset()
// ─────────────────────────────────────────────────────────────────────────────

import { useState } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import "./App.css";

export default function App(){
  // view: which screen to show ("upload" or "chat")
  const [view, setView] = useState("upload");
  // activeDocId: the doc_id of the currently loaded document.
  // null when on the upload screen, set when a document is successfully ingested.
  const [activeDocID, setActiveDocID] = useState(null);

  /**
   * Called by Upload when ingestion completes.
   * Transitions from upload view to chat view.
   */
  function handleUploadSuccess(docID){
    setActiveDocID(docID);
    setView("chat");
  }

  /**
   * Called by Chat when the user clicks "Upload New Document".
   * Transitions back to upload view and clears the active document.
   */
  function handleReset(){
    setActiveDocID(null);
    setView("upload");
  }

  // Conditional rendering: show Upload or Chat based on view state.
  // Only one component renders at a time — the other is unmounted entirely.
  // WHY NOT USE REACT ROUTER:
  // Our app only has two "pages" and no URL-based navigation needed.
  // Adding React Router for two views would be over-engineering. Simple
  // state-based conditional rendering is the right call here.

  return (
    <div className="app">
      {view === "upload" && (
        <Upload onUploadSuccess={handleUploadSuccess} />
      )}
      {view === "chat" && (
        <Chat docID={activeDocID} onReset={handleReset} />
      )}
    </div>
  );
}