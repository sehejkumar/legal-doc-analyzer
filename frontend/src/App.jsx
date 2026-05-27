// App.jsx

import { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import "./App.css";

function makeWelcomeMessage(docId) {
  return {
    role: "assistant",
    content: `Document loaded: **${docId}**\n\nAsk me anything about this document. I'll answer only from its contents and cite my sources.`,
    sources: [],
  };
}

// -----------------------------------------------------------------------------
// Main Application Shell
// -----------------------------------------------------------------------------

/**
 * Root component managing global layout synchronization, indexed document records,
 * view toggles, and multi-document conversation history arrays.
 *
 * Design Choice:
 * Conversation histories are structured as a key-value object indexed by `docId` 
 * directly at the root state level. This keeps data persistent when a user switches 
 * documents, preventing chat logs from being lost when the UI re-renders different views.
 *
 * @returns {JSX.Element} Application shell container with conditional sidebar and view modules.
 */
export default function App() {
  const [documents, setDocuments] = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [view, setView] = useState("loading");
  const [chatHistories, setChatHistories] = useState({});

  // Sync index records from database storage on application initialization
  useEffect(() => {
    async function init() {
      try {
        const { fetchDocuments } = await import("./api");
        const docs = await fetchDocuments();
        setDocuments(docs);
        
        if (docs.length > 0) {
          const firstDoc = docs[0];
          setActiveDocId(firstDoc);
          
          const initialHistories = {};
          docs.forEach(docId => {
            initialHistories[docId] = [makeWelcomeMessage(docId)];
          });
          setChatHistories(initialHistories);
          setView("chat");
        } else {
          setView("upload");
        }
      } catch {
        setView("upload");
      }
    }
    init();
  }, []);

  function handleUploadSuccess(docId) {
    setDocuments(prev => [...new Set([...prev, docId])]);
    setActiveDocId(docId);
    setChatHistories(prev => ({
      ...prev,
      [docId]: [makeWelcomeMessage(docId)],
    }));
    setView("chat");
  }

  function handleSelectDoc(docId) {
    setActiveDocId(docId);
    setView("chat");
  }

  function handleDeleteDoc(docId) {
    const remaining = documents.filter(d => d !== docId);
    setDocuments(remaining);

    setChatHistories(prev => {
      const next = { ...prev };
      delete next[docId];
      return next;
    });

    // Fall back to the next available index file if the active document is deleted
    if (activeDocId === docId) {
      if (remaining.length > 0) {
        setActiveDocId(remaining[0]);
        setView("chat");
      } else {
        setActiveDocId(null);
        setView("upload");
      }
    }
  }

  function handleHistoryUpdate(docId, updater) {
    setChatHistories(prev => ({
      ...prev,
      [docId]: updater(prev[docId] || []),
    }));
  }

  function handleNewUpload() {
    setView("upload");
  }

  if (view === "loading") {
    return (
      <div className="app-loading">
        <div className="loading-spinner" />
      </div>
    );
  }

  const showSidebar = documents.length > 0 || view === "chat";

  return (
    <div className="app-shell">
      {showSidebar && (
        <Sidebar
          documents={documents}
          activeDocId={activeDocId}
          onSelect={handleSelectDoc}
          onDelete={handleDeleteDoc}
          onNewUpload={handleNewUpload}
        />
      )}
      <main className={`app-main ${!showSidebar ? "app-main--full" : ""}`}>
        {view === "upload" && (
          <Upload onUploadSuccess={handleUploadSuccess} />
        )}
        {view === "chat" && activeDocId && (
          /* Design Choice:
             Passing `activeDocId` as a React `key` property forces a clean virtual DOM unmount
             and remount sequence whenever the selected file changes. This completely flushes 
             internal loading or drafting inputs from the child component state, preventing text 
             or stream bleed from leaking into the newly selected document view. */
          <Chat
            key={activeDocId}
            docId={activeDocId}
            messages={chatHistories[activeDocId] || [makeWelcomeMessage(activeDocId)]}
            onHistoryUpdate={(updater) => handleHistoryUpdate(activeDocId, updater)}
          />
        )}
      </main>
    </div>
  );
}