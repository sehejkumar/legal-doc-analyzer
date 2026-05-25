//Upload.jsx
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Lets the user select a PDF file and upload it to the backend.
//          Shows upload progress states and surfaces errors clearly.
//          Calls onUploadSuccess(docId) when ingestion completes so the parent
//          (App.jsx) can switch to the chat view.
// ─────────────────────────────────────────────────────────────────────────────
import { useState, useRef } from "react";
import { uploadPDF } from "../api";

/**
 * PROPS:
 *   onUploadSuccess : (docId: string) => void
 *     Called when the PDF has been fully ingested. Receives the doc_id
 *     (filename) so App.jsx can pass it to the Chat component.
 */

export default function Upload({onUploadSuccess}){
    // ── State ─────────────────────────────────────────────────────────────────
    // useState returns [currentValue, setterFunction].
    // When the setter is called, React re-renders the component with the new value.
    
    // The File object selected by the user (null = nothing selected yet)
    const [selectedFile, setSelectedFile] = useState(null);

    // Upload lifecycle state:
    // "idle"     → nothing happening
    // "uploading" → fetch in progress, show spinner
    // "success"  → ingestion complete
    // "error"    → something went wrong
    const [status, setStatus] = useState("idle");

    //Error message string (only shown when status == "error")
    const [errorMessage, setErrorMessage] = useState("");

    // useRef gives us a reference to the hidden <input type="file"> DOM element
    // so we can trigger a click on it from our custom-styled button.
    // WHY useRef INSTEAD OF document.getElementById:
    // React manages the DOM — directly querying it bypasses React's model and
    // can cause bugs. useRef is the React-idiomatic way to access DOM nodes.
    const fileInputRef = useRef(null);

    // ── Handlers ──────────────────────────────────────────────────────────────

    /**
     * Called when the user selects a file via the file picker.
     * e.target.files is a FileList — we take the first item.
     */
    function handleFileChange(e){
        const file = e.target.files[0]
        if (!file) return;

        // Client-side validation before hitting the server.
        // Catches the wrong file type immediately rather than after an upload.
        // The backend also validates, but failing fast in the UI is better UX.
        if(file.type !== "application/pdf"){
            setStatus("error");
            setErrorMessage("Please select a PDF file!");
            return;
        }

        setSelectedFile(file);
        setStatus("idle");
        setErrorMessage("");
    }

    /**
   * Triggers the hidden file input when the user clicks our custom button.
   * This lets us style the upload button however we want while still using
   * the browser's native file picker.
   */
    function handleSelectClick(){
        fileInputRef.current.click();
    }

    /**
   * Uploads the selected file to the backend.
   * Manages status transitions: idle → uploading → success/error
   */
    async function handleUpload(){
        if(!selectedFile) return;
        setStatus("uploading");
        setErrorMessage("");
        try{
            // uploadPdf() from api.js handles the FormData construction and fetch call.
            // It returns { doc_id, chunks_stored, status } on success.
            const result = await uploadPDF(selectedFile);
            setStatus("success");
            // Notify the parent component. After a short delay for UX (so the user
            // sees the success state before the view changes), switch to chat.
            setTimeout(()=>{
                onUploadSuccess(result.docID);
            },100);
        }catch (err){
            setStatus("error")
            setErrorMessage(err.message);
        }
    }
    // ── Render ────────────────────────────────────────────────────────────────
    // JSX looks like HTML but it's JavaScript. Key differences:
    //   class → className  (class is a reserved word in JS)
    //   onclick → onClick  (camelCase event handlers)
    //   style={{ }} → double braces: outer = JSX expression, inner = JS object
    //
    // Conditional rendering: {condition && <Element />} renders Element only
    // when condition is truthy. This replaces if/else in JSX.
    
    return(
        <div className="upload-container">
            <div className="upload-card">
                {/*Header*/}
                <div className="upload-header">
                    <h1 className="upload-title">Legal Document Analyzer</h1>
                    <p className="upload-subtitle">
                        Upload a PDF contract or legal documentand ask question about it.
                        The AI answers only from the document - no hallucinations!
                    </p>
                </div>
                {/* File selection area */}
                <div className="upload-dropzone" onClick={handleSelectClick}>
                    {/* Hidden native file input — triggered programmatically */}
                    <input type="file" accept=".pdf" ref={fileInputRef} onChange={handleFileChange} style={{display: "none"}} />
                    {/* Icon */}
                    <div className="upload-icon">📄</div>

                    {selectedFile ? (
                        // Show selected filename
                        <div className="upload-selected">
                        <p className="upload-filename">{selectedFile.name}</p>
                        <p className="upload-filesize">
                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                        </div>
                    ) : (
                        // Prompt to select a file
                        <div className="upload-prompt">
                            <p className="upload-prompt-main">Click to select a PDF</p>
                            <p className="upload-prompt-sub">or drag and drop</p>
                        </div>
                    )}
                </div>
                {/* Error message */}
                {status === "error" && (
                    <div className="upload-error">
                        ⚠️ {errorMessage}
                    </div>
                )}

                {/* Upload button */}
                <button
                    className={`upload-btn ${status === "uploading" ? "uploading" : ""}`}
                    onClick={handleUpload}
                    disabled={!selectedFile || status === "uploading" || status === "success"}
                >
                    {status === "idle" && "Upload & Analyze"}
                    {status === "uploading" && "Uploading & Indexing"}
                    {status === "success" && "✓ Done! Opening chat..."}
                    {status === "error" && "Try Again"}
                </button>

                {/* Processing note — shown during upload */}
                {status === "uploading" && (
                    <p className="upload-processing-note">
                        Extracting text, generating embeddings, and indexing your document.
                        This takes 10–30 seconds for the first upload.
                    </p>    
                )}
            </div>
        </div>
    );
}

