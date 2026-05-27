# backend/Dockerfile

# -----------------------------------------------------------------------------
# Base Image & Environment Configuration
# -----------------------------------------------------------------------------

# Use the official slim variant to keep the base layer footprint minimal (~130MB).
# Version is explicitly pinned to guarantee reproducible builds across environment shifts.
FROM python:3.11-slim

# Define the isolated application execution workspace directory context
WORKDIR /app


# -----------------------------------------------------------------------------
# System Dependencies (Pre-pip installation steps)
# -----------------------------------------------------------------------------

# Install low-level system binaries required by text parsers.
# Design Choice:
# Combining the update, install, and cache purging commands within a single RUN directive
# prevents unwanted layer accumulation, minimizing the final compiled image volume.
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    pkg-config \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# Python Dependencies & Layer Caching Optimization
# -----------------------------------------------------------------------------

# Design Choice:
# requirements.txt is staged and evaluated independently before moving source code.
# This ensures that subsequent modifications to logic files skip heavy pip installation 
# cycles by pulling directly from the local Docker layer build cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Runtime Cold-Start Optimizations
# -----------------------------------------------------------------------------

# Design Choice:
# Explicitly initialize and cash the embedding weights directly during the build execution phase.
# Baking the ~90MB neural network payload straight into the image layers avoids runtime download
# bottlenecks that would otherwise trip cloud provider platform health check limits.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-MiniLM-L3-v2')"


# -----------------------------------------------------------------------------
# Application Source Assembly & Directory Layout
# -----------------------------------------------------------------------------

# Ingest all tracking source components from the localized build execution context scope
COPY . .

# Explicitly provision directories intended for dynamic file cache modifications
RUN mkdir -p uploads chroma_db


# -----------------------------------------------------------------------------
# Networking & Application Invocations
# -----------------------------------------------------------------------------

# Informative port tracking reference configuration
EXPOSE 8000

# Design Choice:
# The process entry point uses shell expansion to dynamically read environment ports assigned at deployment.
# We explicitly bind to 0.0.0.0 to instruct the container to listen to external interface routing hooks 
# instead of isolating traffic loops exclusively inside container-level loops.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}