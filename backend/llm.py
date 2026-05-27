# llm.py

import os
from groq import Groq
from dotenv import load_dotenv
from typing import Generator

# -----------------------------------------------------------------------------
# Configuration & Initialization
# -----------------------------------------------------------------------------

# Load environment variables from local .env file to prevent API key leakage in source control.
load_dotenv()

# The Groq client automatically targets the GROQ_API_KEY environment variable.
# It features an OpenAI-compatible SDK pattern, making it straightforward to swap
# providers if needed later.
client = Groq()

# Llama 3.3 70B model identifier used for core text generation.
MODEL = "llama-3.3-70b-versatile"


# -----------------------------------------------------------------------------
# Prompt Construction
# -----------------------------------------------------------------------------

def buildPrompt(query: str, chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    """
    Assembles a stateless, structured messages array for the Chat Completions API,
    combining a strict system context, conversation history, and the new query.

    Design Choice:
    The LLM API is entirely stateless. To simulate continuous conversation memory, 
    prior message exchanges must be appended in order before the current user question.
    Additionally, document chunks are inserted directly into the system instructions.
    This anchors model behavior more reliably than user-space injections, ensuring 
    the model adheres tightly to the source text.

    Parameters:
        query (str): The incoming user question.
        chunks (list[dict]): Retrieved document sections containing 'text' and 'chunkIndex'.
        history (list[dict] | None): Past chat turns formatted as {'role': ..., 'content': ...}.

    Returns:
        list[dict]: Ordered payload ready for transmission to the model endpoint.
    """
    # Group sections with visible structural boundaries so the model can isolate text blocks
    contextParts = []
    for i, currChunk in enumerate(chunks, start=1):
        header = f"[Excerpt {i} - Chunk {currChunk['chunkIndex']}]"
        contextParts.append(f"{header}\n{currChunk['text']}")

    context = "\n\n---\n\n".join(contextParts)

    # Core system instructions outlining precision rules and factual alignment constraints.
    systemPrompt = f"""You are a precise legal document assistant. Your job is to answer questions about the legal document the user has uploaded.
 
RULES YOU MUST FOLLOW:
1. ONLY use the document excerpts provided below to answer questions.
2. Do NOT use any outside knowledge, even if you are confident in it.
3. If the answer cannot be found in the excerpts, say: "I could not find information about this in the provided document."
4. When you answer, cite which excerpt(s) you used (e.g. "According to Excerpt 2...").
5. Be concise and precise. Restrict your answer exclusively to what is explicitly stated. Do NOT interpolate, extrapolate, or infer any terms, "implications," or logical conclusions not directly written in the text.
6. If a question requires a legal opinion or interpretation, clarify that you can only relay exactly what the document states.
7. You have access to the conversation history. You may reference earlier questions and answers.
 
DOCUMENT EXCERPTS:
{context}"""
    
    # Initialize message log with high-priority system context
    messages = [{"role": "system", "content": systemPrompt}]
    
    if history:
        messages.extend(history)

    # The current message must always be the final element in the array
    messages.append({"role": "user", "content": query})

    return messages


# -----------------------------------------------------------------------------
# Streaming Generation Pipeline
# -----------------------------------------------------------------------------

def streamResponse(query: str, chunks: list[dict], history: list[dict] | None = None) -> Generator[str, None, None]:
    """
    Executes a streaming request against the Groq API, yielding tokens as they 
    become available to support real-time token rendering.

    Design Choice:
    Utilizes a Python generator (`yield`) to bridge the API and FastAPI endpoints.
    A temperature setting of 0.1 is enforced to optimize for consistency and factual
    correctness, effectively disabling creative variance that could lead to hallucinated
    legal terms.

    Parameters:
        query (str): The question to pass to the processing pipeline.
        chunks (list[dict]): The semantic context retrieved from vector storage.
        history (list[dict] | None): Optional list tracking historical chat messages.

    Yields:
        str: Individual text tokens from the response chunk delta.
    """
    messages = buildPrompt(query, chunks, history)

    # Run completion with stream tracking enabled
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True
    )

    # Extract raw content strings from the API chunk stream structure
    for currChunk in completion:
        currToken = currChunk.choices[0].delta.content
        if currToken:
            yield currToken


# -----------------------------------------------------------------------------
# Synchronous Testing Utility
# -----------------------------------------------------------------------------

def getResponse(query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """
    Executes a standard synchronous blocking request against the Groq API.

    Design Choice:
    Maintained primarily as a developer testing utility. It simplifies evaluation 
    tasks via terminal scripts or internal Swagger interfaces (`/docs`) where 
    streaming overhead isn't needed.

    Parameters:
        query (str): User question.
        chunks (list[dict]): Document context data blocks.
        history (list[dict] | None): Pre-existing conversation log components.

    Returns:
        str: The full generated text block as a unified string payload.
    """
    messages = buildPrompt(query, chunks, history)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=False
    )
    return completion.choices[0].message.content