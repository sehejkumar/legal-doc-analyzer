//chat.jsx
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: The main chat interface. Displays conversation history, accepts
//          user questions, and streams the AI's response token by token.
//
// This is the most complex component — it manages:
//   - Conversation history (array of {role, content} objects)
//   - Streaming state (building the assistant message in real time)
//   - Input state (controlled input field)
//   - Auto-scrolling to the latest message
// ─────────────────────────────────────────────────────────────────────────────

import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api";

/**
 * PROPS:
 *   docId    : string  →  the document ID to query against (from Upload)
 *   onReset  : () => void  →  called when user clicks "Upload New Document"
 */

export default function Chat({docID, onReset}){
    // ── State ─────────────────────────────────────────────────────────────────

  // messages: the full conversation history.
  // Each message is { role: "user" | "assistant", content: string }
  // We initialize with a welcome message so the chat isn't empty on load.

    const [messages, setMessages] = useState([
    {
        role: "assistant",
        content: `Document loaded: **${docID}**\n\nAsk me about anything in this document I'll answer only from its contents.`,
    },
    ]);

  // The current value of the text input (controlled component pattern).
  // "Controlled" means React state is the source of truth for the input value,
  // not the DOM. onChange keeps state in sync with what the user types.
    const [input, setInput] = useState("");
    // isStreaming: true while the LLM is generating a response.
  // Used to disable the input and show a loading indicator.
    const [isStreaming, setIsStreaming] = useState(false);

  // Refs for DOM access:
  // messagesEndRef: an invisible div at the bottom of the message list.
  // We scroll it into view whenever messages update = auto-scroll to latest.
    const messagesEndRef = useRef(null);
    // inputRef: reference to the text input so we can refocus it after sending.
    const inputRef = useRef(null);

    // ── Effects ───────────────────────────────────────────────────────────────
 
  // useEffect runs after every render where the dependency array changes.
  // This one runs whenever `messages` changes — i.e. after every new message
  // or token append. It scrolls the chat to the bottom automatically.
  //
  // WHY useEffect AND NOT JUST CALL scrollIntoView DIRECTLY:
  // At the point where we update state, the DOM hasn't re-rendered yet.
  // useEffect runs after React has committed the DOM update, so the new
  // message element exists in the DOM when we try to scroll to it.
    useEffect(()=>{
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    },[messages]);

    // ── Handlers ──────────────────────────────────────────────────────────────
    async function handleSend(){
        // Guard: don't send empty messages or while already streaming
        const query = input.trim()
        if(!query || isStreaming) return;
        // ── Add user message to history ─────────────────────────────────────────
    // We use the functional form of setState: setMessages(prev => [...])
    // WHY FUNCTIONAL FORM:
    // React batches state updates. If we used setMessages([...messages, newMsg])
    // and messages hadn't updated yet from a previous setState, we'd be spreading
    // a stale version of messages. The functional form always receives the latest
    // state as its argument, preventing stale closure bugs.
        const userMessage = {role: 'user', content: query};
        setMessages(prev=>[...prev,userMessage]);
        setInput("");
        setIsStreaming(true);
        
        // ── Add empty assistant message as a placeholder ────────────────────────
    // We immediately add an empty assistant message to the history.
    // As tokens stream in, we'll update this message's content in place.
    // This means the user sees "▋" (the typing cursor) immediately, then
    // text appearing — rather than silence followed by a sudden full response.
    //
    // The index of this placeholder is messages.length + 1 (user message is +0,
    // assistant placeholder is +1). But since we can't rely on state having
    // updated synchronously, we track the assistant message by always updating
    // the last message in the array.
        const assistantPlaceholder = {role: "assistant", content:""};
        setMessages(prev=>[...prev, assistantPlaceholder]);

    // ── Stream the response ─────────────────────────────────────────────────
    // askQuestion() calls the backend and invokes our callbacks as tokens arrive.
        await askQuestion(
            query,
            docID,
            // onToken: called with each text token (typically 1-4 words)
      // We update the last message in the array (the assistant placeholder)
      // by appending the new token to its content.
      //
      // prev.slice(0, -1) = all messages except the last one
      // lastMsg = the current last message (our placeholder)
      // { ...lastMsg, content: lastMsg.content + token } = updated copy
      // [...prev.slice(0, -1), updatedLast] = full array with update applied
      //
      // This is the immutable update pattern — we never mutate state directly,
      // we always create a new array. React requires this to detect changes.
            (token) => {
                setMessages(prev => {
                    const lastMessage = prev[prev.length - 1];
                    const updateLastMessage = { ...lastMessage, content: lastMessage.content + token};
                    return [...prev.slice(0,-1), updateLastMessage];
                });
            },
            // onDone: stream finished, re-enable input
            () => {
                setIsStreaming(false);
                // Refocus the input field so the user can immediately type another question
                setTimeout(()=>inputRef.current?.focus(),50);
            },
            // onError: something went wrong — show error message in chat
            (errorMessage) => {
                setMessages(prev =>{
                    const lastMessage = prev[prev.length-1];
                    const updatedLast = {
                        ...lastMessage,
                        content: `⚠️ Error: ${errorMessage}`,
                        isError: true,
                    };
                    return [...prev.slice(0,-1),updatedLast];
                });
                setIsStreaming(false);
            }
        )
    }

    /**
   * Allows submitting by pressing Enter (but not Shift+Enter, which inserts
   * a newline). This matches the UX convention of most chat applications.
   */
    function handleKeyDown(e){
        if (e.key === "Enter" && !e.shiftKey){
            e.preventDefault(); //prevent newline being inserted
            handleSend();
        }
    }

    /**
   * Renders message content with basic markdown-like formatting.
   * Handles **bold** and line breaks. For a production app you'd use
   * a library like react-markdown, but this keeps dependencies minimal.
   */
    function renderContent(content){
        // Split on **text** pattern and render bold spans
        const parts = content.split(/(\*\*[^*]+\*\*)/g);
        return parts.map((part,i) =>{
            if(part.startsWith("**") && part.endsWith("**")){
                return <strong key={i}>{part.slice(2,-2)}</strong>
            }
            //Replace \n with <br> for line breaks
            return part.split("\n").map((line,j,arr) =>(
                <span key={`${i}-${j}`}>
                    {line}
                    {j < arr.length - 1 && <br />}
                </span>
            ));
        });
    }

    // ── Render ────────────────────────────────────────────────────────────────
    return (
    <div className="chat-container">

      {/* Header bar */}
        <div className="chat-header">
        <div className="chat-header-info">
            <span className="chat-doc-label">Document:</span>
            <span className="chat-doc-name">{docID}</span>
        </div>
        <button className="chat-reset-btn" onClick={onReset}>
            Upload New Document
        </button>
        </div>

      {/* Messages list */}
        <div className="chat-messages">
        {messages.map((msg, index) => (
            <div
            key={index}
            className={`chat-message ${msg.role} ${msg.isError ? "error" : ""}`}
            >
            {/* Role label */}
            <div className="chat-message-label">
                {msg.role === "user" ? "You" : "AI"}
            </div>

            {/* Message content */}
            <div className="chat-message-content">
                {renderContent(msg.content)}

              {/* Blinking cursor shown on the last message while streaming */}
                {isStreaming &&
                index === messages.length - 1 &&
                msg.role === "assistant" && (
                    <span className="chat-cursor">▋</span>
                )}
            </div>
            </div>
        ))}

        {/* Invisible anchor div at the bottom — scrolled into view on update */}
        <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="chat-input-area">
        <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the document..."
            rows={2}
            disabled={isStreaming}
        />
        <button
            className={`chat-send-btn ${isStreaming ? "streaming" : ""}`}
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
        >
            {isStreaming ? "..." : "Send"}
        </button>
        </div>

      {/* Hint text */}
        <p className="chat-hint">
        Press Enter to send · Shift+Enter for new line · AI answers only from the document
        </p>

    </div>
    );
}