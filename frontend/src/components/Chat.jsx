// Chat.jsx

import { useRef, useEffect, useState } from "react";
import { askQuestion } from "../api";

// -----------------------------------------------------------------------------
// Interactive Chat Interface Component
// -----------------------------------------------------------------------------

/**
 * Handles message stream state updates, input textarea auto-scaling, 
 * conversational scrolling layout locks, and localized transcript exports.
 *
 * Design Choice:
 * State updates for chunk sources and incoming stream tokens target the final array 
 * entry in place. This localized buffer update eliminates message block stutter or 
 * structural shifting during intensive real-time token rendering passes.
 *
 * @param {string} docId - Namespace targeting the active document index.
 * @param {Array<Object>} messages - Parsed collection of dialog entries to display.
 * @param {Function} onHistoryUpdate - Global history log state modifier dispatch callback.
 * @returns {JSX.Element} Fully active, auto-scrolling chat log pane window module.
 */
export default function Chat({ docId, messages, onHistoryUpdate }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll anchor adjustment executed on message feed additions
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle auto-expanding height resizing logic on text area insertions
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  async function handleSend() {
    const query = input.trim();
    if (!query || isStreaming) return;

    // Slice out the initial welcome greeting to send clean conversational contexts
    const history = messages
      .slice(1)
      .map(m => ({ role: m.role, content: m.content }));

    const userMsg = { role: "user", content: query, sources: [] };
    onHistoryUpdate(prev => [...prev, userMsg]);

    setInput("");
    setIsStreaming(true);

    // Initialize an empty streaming frame placeholder
    const placeholder = { role: "assistant", content: "", sources: [] };
    onHistoryUpdate(prev => [...prev, placeholder]);

    await askQuestion(
      query,
      docId,
      history,

      // Handle structural citation injections
      (sources) => {
        onHistoryUpdate(prev => {
          const last = prev[prev.length - 1];
          return [...prev.slice(0, -1), { ...last, sources }];
        });
      },

      // Append raw message deltas onto the tracking layout
      (token) => {
        onHistoryUpdate(prev => {
          const last = prev[prev.length - 1];
          return [...prev.slice(0, -1), { ...last, content: last.content + token }];
        });
      },

      // Pipeline resolution cleanup hook
      () => {
        setIsStreaming(false);
        setTimeout(() => inputRef.current?.focus(), 50);
      },

      // Invalidate stream state context targets on network crash alerts
      (errMsg) => {
        onHistoryUpdate(prev => {
          const last = prev[prev.length - 1];
          return [
            ...prev.slice(0, -1),
            { ...last, content: `Error: ${errMsg}`, isError: true },
          ];
        });
        setIsStreaming(false);
      }
    );
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleDownload() {
    const lines = messages.map(m => {
      const role = m.role === "user" ? "You" : "AI";
      const sources = m.sources?.length
        ? `\n[Sources: ${m.sources.map(s => `Chunk ${s.chunkIndex}`).join(", ")}]`
        : "";
      return `${role}:\n${m.content}${sources}`;
    });

    const text = [
      "Legal Document Analyzer — Chat Transcript",
      `Document: ${docId}`,
      `Date: ${new Date().toLocaleDateString()}`,
      "",
      "─".repeat(50),
      "",
      lines.join("\n\n"),
    ].join("\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `transcript-${docId.replace(".pdf", "")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function renderContent(content) {
    return content.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part.split("\n").map((line, j, arr) => (
        <span key={`${i}-${j}`}>{line}{j < arr.length - 1 && <br />}</span>
      ));
    });
  }

  return (
    <div className="chat-page">
      <header className="chat-topbar">
        <div className="chat-topbar-left">
          <div className="chat-doc-badge">
            <span className="chat-doc-badge-icon">📄</span>
            <span className="chat-doc-badge-name" title={docId}>{docId}</span>
          </div>
        </div>
        <div className="chat-topbar-right">
          <button className="topbar-btn" onClick={handleDownload} title="Download transcript">
            ↓ Export
          </button>
        </div>
      </header>

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chat-msg chat-msg--${msg.role} ${msg.isError ? "chat-msg--error" : ""}`}
          >
            <div className="chat-msg-avatar">
              {msg.role === "user" ? "You" : "⚖"}
            </div>
            <div className="chat-msg-body">
              <div className="chat-msg-content">
                {renderContent(msg.content)}
                {isStreaming && index === messages.length - 1 && msg.role === "assistant" && (
                  <span className="chat-cursor">▋</span>
                )}
              </div>
              
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <div className="chat-sources">
                  <span className="chat-sources-label">Sources</span>
                  {msg.sources.map((src, i) => (
                    <span key={i} className="chat-source-chip">
                      Excerpt {i + 1} · Chunk {src.chunkIndex}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-bar">
        <div className="chat-input-wrap">
          <textarea
            ref={el => { textareaRef.current = el; inputRef.current = el; }}
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the document…"
            rows={1}
            disabled={isStreaming}
          />
          <button
            className={`chat-send ${isStreaming ? "chat-send--streaming" : ""}`}
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
          >
            {isStreaming ? <span className="send-spinner" /> : <span>↑</span>}
          </button>
        </div>
        <p className="chat-hint">
          Enter to send · Shift+Enter for new line · Answers grounded in document only
        </p>
      </div>
    </div>
  );
}