#ingest,py

'''
Purpose: take a pdf file, extract its text, split it into overlapping chunks,
convert each chunk into a vector embedding, and store everything in ChromaDB so 
it can be searched later.

This fule runs once per uploaded pdf ("indexing" phase). The "querying" phase
will search what we store here.
'''

import pdfplumber #extracts text from PDF files
import chromadb #Vector Database - stores and searches embeddings
from sentence_transformers import SentenceTransformer #converts text -> vectors

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load the embedding model
# ─────────────────────────────────────────────────────────────────────────────

'''
We load the model once at a module import time (when Python first reads this 
file), not inside a function. If we loaded it inside a function, it would 
re-download and re-initialize the model (~80MB neural net) on every single PDF
upload and would take 3-5 seconds each time. Loading once at startup time costs
only on first run, then it is in memory for all future calls.
'''

'''
What is "all-MiniLM-L6-v2":
 - A small but powerfful sentence embedding model.
  - MiniLM: compressed version of microsoft's MiniLM architecture
  - L6: 6 transformer layers
  - v2: version 2

It produces 384-dimensional vectors and runs on CPU in ~20ms per sentence.
First run downloads ~90MB from Hugging Face. Cached locally after that
'''

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Initialize ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

'''
ChromaDB has two modes:
- EphemeraClient() -> stores data only in RAM. Gone when program stops.
- PersistentClient() -> writes data to disk. Survives restarts

We will use PersistentClient() so if I have to restart the FASTAPI server, all
the previously indexed documents are still available.

path="./chroma_db" means ChromaDB will create a folder called "chroma_db" in the
backend directory and write its files there. This folder is what you would add
to gitignore (contains GB of vector data)
'''

client = chromadb.PersistentClient(path="./chroma_db")

'''
ChromaDB orgabizes data into "collections" (similar to tables in SQL). Each
collection holds a set of (text, vector, metadata) tuples. We use one
collection called "documents".

get_or_create_collection: 
- if "documents" already exists, use the existing one
- if it does not exist yet, creare a fresh one
this prevents errors on restart and avoids duplication of data
'''

collection = client.get_or_create_collection("documents")