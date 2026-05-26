# Legal Document Analyzer

An AI-powered web app that lets you upload any legal PDF and have a conversation with it. Ask questions in plain English — the AI answers based solely on the document's contents using a RAG (Retrieval-Augmented Generation) pipeline.

**Fully free to run.** No paid APIs, no Docker, no cloud setup required.

---

## What It Does

- Upload a legal PDF (contract, NDA, terms of service, etc.)
- Ask natural-language questions: *"What is the termination clause?"*, *"Does this contract have a penalty for late payment?"*
- The AI retrieves the most relevant sections and answers only from those — no hallucination from outside knowledge
- Responses stream token-by-token in real time

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite (JavaScript + SWC) |
| Backend | Python · FastAPI · Uvicorn |
| PDF Parsing | pdfplumber |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — local, free |
| Vector DB | ChromaDB — local, no server needed |
| LLM | Llama 3.3 70B via Groq API (free tier) |

---

## How It Works (RAG Pipeline)

**On upload:**
1. `pdfplumber` extracts text from the PDF
2. Text is split into overlapping 500-word chunks (50-word overlap)
3. `sentence-transformers` converts each chunk into a 384-dimensional vector embedding
4. Chunks + vectors are stored in ChromaDB on disk

**On every question:**
1. The question is embedded into the same vector space
2. ChromaDB finds the 4 most semantically similar chunks (cosine similarity)
3. Those chunks are injected into a grounded system prompt
4. Llama 3.3 70B via Groq generates an answer, streamed token-by-token to the UI

---

## Project Structure

```
legal-doc-analyzer/
├── backend/
│   ├── main.py          # FastAPI app — 4 endpoints
│   ├── ingest.py        # PDF parsing, chunking, embedding, ChromaDB storage
│   ├── retriever.py     # Semantic search against ChromaDB
│   ├── llm.py           # Prompt assembly + Groq API streaming call
│   ├── requirements.txt # Python dependencies
│   └── .env             # GROQ_API_KEY (not committed)
└── frontend/
    ├── src/
    │   ├── App.jsx          # Root component (upload/chat state machine)
    │   ├── App.css          # All styles
    │   ├── api.js           # Service layer — all fetch calls
    │   └── components/
    │       ├── Upload.jsx   # PDF upload screen
    │       └── Chat.jsx     # Chat interface with streaming
    └── package.json
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repo

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
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

### 4. Run the app

Open two terminals:

**Terminal 1 — Backend:**
```bash
cd backend
.venv\Scripts\activate
uvicorn main:app --reload
# Running at http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# Running at http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/documents` | List all indexed document IDs |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question (streaming response) |

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Key Design Decisions

**Why RAG instead of fine-tuning?**
Fine-tuning is expensive and slow. RAG injects the relevant document text directly into the prompt at query time — cheaper, faster, and works with any document without retraining.

**Why `sentence-transformers` instead of an embedding API?**
Runs locally, completely free, no rate limits, no data leaves your machine. The `all-MiniLM-L6-v2` model is 90MB and produces high-quality 384-dim embeddings on CPU.

**Why Groq instead of OpenAI?**
Groq's free tier requires no credit card. Its LPU hardware makes Llama 3.3 70B inference faster than GPT-4o in practice. The OpenAI-compatible SDK means switching later is a one-line change.

**Why overlapping chunks?**
Without overlap, sentences that land on chunk boundaries lose context. 50-word overlap ensures every idea appears in full context in at least one chunk.

**Why low temperature (0.1)?**
Legal document Q&A requires factual, consistent answers. Low temperature keeps the model grounded to the provided context rather than creatively paraphrasing legal text.

---

## Limitations

- Scanned PDFs (image-only, no text layer) are not supported — OCR not implemented
- Very large PDFs (100+ pages) may take 30-60 seconds to index on first upload
- Free Groq tier has rate limits — for heavy use, consider caching responses

---

## License

MIT