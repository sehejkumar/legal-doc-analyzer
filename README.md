# LexAI — Legal Document Analyzer

An AI-powered web app for querying legal documents in plain language. Upload a PDF contract, NDA, lease, or terms of service and ask questions — the AI answers exclusively from the document's contents using a RAG (Retrieval-Augmented Generation) pipeline with full source citations.

**Entirely free to run.** No paid APIs, no Docker, no cloud infrastructure required.

---

## Features

- **Semantic Q&A** — ask natural-language questions, get grounded answers
- **Source citations** — every answer shows which document excerpts it drew from
- **Conversation memory** — follow-up questions reference prior answers
- **Multi-document support** — upload and switch between multiple documents via sidebar
- **Export transcript** — download the full conversation as a `.txt` file
- **Streaming responses** — answers appear token-by-token in real time

---

## How It Works (RAG Pipeline)

**On upload — indexing phase:**
1. `pdfplumber` extracts raw text from the PDF
2. Text is split into overlapping 500-word chunks (50-word overlap to preserve context at boundaries)
3. `sentence-transformers` (`all-MiniLM-L6-v2`) converts each chunk into a 384-dimensional vector embedding — runs locally, no API call
4. Chunks, vectors, and metadata are stored in ChromaDB on disk

**On every question — querying phase:**
1. The question is embedded into the same vector space
2. ChromaDB finds the 4 most semantically similar chunks via cosine similarity
3. Chunks are injected into a grounded system prompt with strict rules against outside knowledge
4. Llama 3.3 70B via Groq generates an answer, streamed token-by-token to the UI
5. Source metadata (chunk indices) is sent before text begins so citations appear immediately

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React 18 + Vite (JavaScript + SWC) | Fast dev server, modern React, no config |
| Backend | FastAPI + Uvicorn | Async, auto-docs at `/docs`, industry standard for ML APIs |
| PDF Parsing | pdfplumber | Best handling of multi-column legal layouts |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local, free, 384-dim, CPU-friendly |
| Vector DB | ChromaDB | In-process, persistent, no server needed |
| LLM | Llama 3.3 70B via Groq API | Free tier, faster than GPT-4o, OpenAI-compatible SDK |

---

## Project Structure

```
legal-doc-analyzer/
├── .gitignore
├── README.md
├── backend/
│   ├── .env                  ← GROQ_API_KEY (gitignored)
│   ├── .env.example          ← template for .env
│   ├── requirements.txt      ← pinned Python dependencies
│   ├── main.py               ← FastAPI app — 5 endpoints
│   ├── ingest.py             ← PDF parsing, chunking, embedding, storage
│   ├── retriever.py          ← semantic search + document management
│   └── llm.py                ← prompt assembly, conversation memory, Groq streaming
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx               ← root — document list state, view routing
        ├── App.css               ← all styles (DM Serif Display + DM Sans)
        ├── api.js                ← service layer — all fetch calls
        └── components/
            ├── Sidebar.jsx       ← document list, selection, deletion
            ├── Upload.jsx        ← PDF upload with drag-and-drop
            └── Chat.jsx          ← chat UI, streaming, source chips, export
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A free Groq API key from [console.groq.com](https://console.groq.com) — no credit card required

### 1. Clone

```bash
git clone https://github.com/yourusername/legal-doc-analyzer.git
cd legal-doc-analyzer
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and add your Groq API key
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

### 4. Run

Open **two terminals**:

**Terminal 1 — Backend:**
```bash
cd backend
.venv\Scripts\activate        # Windows / source .venv/bin/activate Mac/Linux
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI — test all endpoints here)
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# → http://localhost:5173
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/documents` | List all indexed document IDs |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question with conversation history (streaming) |
| `DELETE` | `/documents/{doc_id}` | Remove a document from the index |

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### POST /ask — Request body

```json
{
  "query": "What is the termination clause?",
  "doc_id": "contract.pdf",
  "history": [
    { "role": "user",      "content": "Who are the parties?" },
    { "role": "assistant", "content": "According to Excerpt 1..." }
  ]
}
```

### POST /ask — Streaming response format

The response is a plain-text stream with two phases:

```
{"sources": [{"chunk_index": 4, "doc_id": "contract.pdf"}, ...]}\n
According to Excerpt 1, either party may terminate...
```

Line 1: JSON metadata with source chunk indices
Line 2+: streaming LLM tokens

---

## Key Design Decisions

**Why RAG instead of fine-tuning?**
Fine-tuning is expensive, slow, and the model still tends to hallucinate specific facts. RAG retrieves the actual document text at query time and injects it into the prompt — cheaper, faster, no retraining, works with any document.

**Why `sentence-transformers` instead of an embedding API?**
Runs locally — completely free, no rate limits, no data sent to a third party. The `all-MiniLM-L6-v2` model (~90MB) produces high-quality 384-dimensional embeddings on CPU in ~20ms per sentence.

**Why overlapping chunks?**
Without overlap, sentences at chunk boundaries lose context. With 50-word overlap, every idea appears in full context in at least one chunk and retrieval accuracy improves noticeably.

**Why Groq instead of OpenAI?**
Groq's free tier requires no credit card. Its LPU hardware makes Llama 3.3 70B inference faster than GPT-4o in practice. The OpenAI-compatible SDK means switching later is a one-line change.

**Why low temperature (0.1)?**
Legal Q&A requires factual, consistent answers. Low temperature keeps the model grounded to the provided context rather than creatively paraphrasing legal text, which risks distorting meaning.

**Why re-inject context on every turn?**
The LLM is stateless — it has no memory between calls. We simulate memory by sending the full conversation history on every request. The document excerpts are re-injected every turn too, ensuring grounding is maintained even on follow-up questions like "how many days notice does that require?"

**Why send source metadata before text in the stream?**
So the frontend can display source chips immediately, before the answer starts streaming in. This reinforces trust — the AI shows its evidence before making its claim.

---

## Limitations

- Scanned PDFs (image-only, no text layer) are not supported — OCR not implemented
- Large PDFs (100+ pages) may take 30–60 seconds to index on first upload
- Groq free tier has rate limits — for heavy use, consider response caching

---

## License

MIT