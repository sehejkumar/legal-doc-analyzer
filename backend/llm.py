#llm.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Take retrieved chunks + user question, assemble a grounded prompt,
#          call the Groq API (Llama 3.3 70B), and stream the response back.
#
# This is the final step in the RAG pipeline before the answer reaches the user.
#
# Data flow:
#   chunks (List[str]) + query (str)
#       → buildPrompt()     → structured messages list
#       → askLLM()          → Groq API call (streaming)
#       → streamResponse()  → yields text tokens one by one
#                           → FastAPI streams these to the frontend
# ─────────────────────────────────────────────────────────────────────────────

import os
from groq import Groq
from dotenv import load_dotenv
from typing import Generator

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load environment variables and initialize Groq client
# ─────────────────────────────────────────────────────────────────────────────

# load_dotenv() reads the .env file in the current directory and loads each
# line (KEY=VALUE) into the process's environment variables.
# After this call, os.environ["GROQ_API_KEY"] will return the key from .env.
#
# WHY NOT HARDCODE THE KEY?
# API keys hardcoded in source code get committed to git and exposed publicly.
# Automated bots scan GitHub for leaked keys within minutes of a push.
# .env + load_dotenv() is the universal standard for local development secrets.
# In production (e.g. Railway, Render), you set environment variables in the
# platform's dashboard instead — load_dotenv() just makes local dev work the
# same way.

load_dotenv()

# Groq() with no arguments automatically looks for the GROQ_API_KEY environment
# variable. This is a common SDK pattern — you don't pass the key explicitly,
# the SDK reads it from the environment. Cleaner and harder to accidentally log.
#
# The Groq client uses the OpenAI-compatible API format. If you ever want to
# switch to OpenAI, the change is:
#   from openai import OpenAI
#   client = OpenAI()   # reads OPENAI_API_KEY from environment
# Everything else in this file stays identical. That's intentional API design.

client = Groq()

# The model string for Llama 3.3 70B on Groq.
# Defined as a constant so changing models is a one-line edit.
# Common alternatives on Groq's free tier:
#   "llama-3.1-8b-instant"   → faster, smaller, less capable
#   "mixtral-8x7b-32768"     → good for long documents (32k context)
#   "gemma2-9b-it"           → Google's Gemma 2, fast and capable

MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Prompt assembly
# ─────────────────────────────────────────────────────────────────────────────

def buildPrompt(query: str, chunks: list[str])->list[dict]:
    '''
    Assembles the retrieved chunks and user query into a structured message
    list for the Groq API.

    WHAT "MESSAGES" ARE:
    The Groq/OpenAI API doesn't take a single string — it takes a list of
    message objects, each with a "role" and "content". The roles are:
      - "system"    : instructions to the model about how to behave
      - "user"      : the human's message
      - "assistant" : the model's previous responses (for multi-turn chat)
    This structure lets the API handle conversation history naturally.
    For now we only have system + user (single turn). In Phase 5 we'll add
    conversation history for multi-turn chat.

    WHAT THE SYSTEM PROMPT DOES:
    The system prompt is the most important part of a RAG implementation.
    It does three things:
      1. Defines the model's persona ("legal document assistant")
      2. Provides the retrieved context (the chunks)
      3. Gives strict grounding instructions ("only use the context below")
 
    Without grounding instruction #3, the model will freely mix its training
    knowledge with the document content — which looks helpful but is actually
    dangerous for legal use cases because the model might "fill in" plausible
    but incorrect legal details from its training data.

    PARAMETERS:
      query  : str       →  the user's question
      chunks : List[str] →  retrieved text chunks from retriever.py
 
    RETURNS:
      List[dict] — the messages array ready to send to the API
    '''

    #Build the context block
    # Join all chunks into one context string, numbered for clarity.
    # Numbering helps the model reference specific sections and makes debugging
    # easier — you can see exactly which chunk informed the answer.

    # The separator "\n\n---\n\n" creates a clear visual boundary between chunks.
    # This matters because the model reads the context as one long string and the
    # boundary helps it distinguish where one chunk ends and another begins.

    contextParts = []
    for i, currChunk in enumerate(chunks,start=1):
        contextParts.append(f"[Excerpt {i}]\n{currChunk}")

    context = "\n\n---\n\n".join(contextParts)

    #Build the system prompt

    # This is the prompt engineering heart of the system.
    #
    # KEY DESIGN DECISIONS:
    #
    # 1. "ONLY use the excerpts provided" — strict grounding. The model is
    #    forbidden from using outside knowledge. For legal documents this is
    #    essential: you never want the model to hallucinate a clause that sounds
    #    plausible but isn't in the actual contract.
    #
    # 2. "If the answer is not in the excerpts, say so explicitly" — graceful
    #    failure. A RAG system that says "I don't see that in this document" is
    #    far more trustworthy than one that fabricates an answer. Users learn to
    #    trust the system because it's honest about its limitations.
    #
    # 3. "cite which excerpt" — traceability. Users can verify the answer
    #    against the source. This is table stakes for legal/professional use.
    #
    # 4. "Be concise and precise" — legal users want clear answers, not hedging.
    #    LLMs tend toward verbosity; this instruction counteracts that.
    #
    # 5. Putting the context INSIDE the system prompt (not the user message) —
    #    this is a design choice. Some implementations put context in the user
    #    message. System prompt context tends to get stronger adherence because
    #    the model treats system instructions as higher priority than user input.

    systemPrompt =f"""You are a precise legal document assistant. Your job is to answer questions about the legal document the user has uploaded.
        
        RULES YOU MUST FOLLOW:
        1. ONLY use the document excerpts provided below to answer questions.
        2. Do NOT use any outside knowledge, even if you are confident in it.
        3. If the answer cannot be found in the excerpts, say: "I could not find information about this in the provided document."
        4. When you answer, cite which excerpt(s) you used (e.g. "According to Excerpt 2...").
        5. Be concise and precise. Avoid unnecessary hedging or filler language.
        6. If a question requires a legal opinion (not just facts from the document), clarify that you can only relay what the document states, not provide legal advice.
        
        DOCUMENT EXCERPTS:
        {context}"""
    
    #Assemble the messages list
    # The API receives a list of message dicts. Order matters:
    # system always comes first, then the conversation history, then the latest
    # user message. The model reads them top to bottom like a conversation log.

    messages = [
        {
            "role": "system",
            "content": systemPrompt,
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LLM call with streaming
# ─────────────────────────────────────────────────────────────────────────────

def streamResponse(query: str, chunks: list[str]) -> Generator[str,None, None]:
    '''
    Builds the prompt, calls Groq with streaming enabled, and yields
    text tokens one by one as they arrive from the API.

    WHAT A GENERATOR IS:
    A generator function uses "yield" instead of "return". When called, it
    doesn't execute immediately — it returns a generator object. Each time
    the caller asks for the next value (via next() or a for loop), the function
    runs until it hits the next yield, pauses, and gives that value back.
 
    WHY USE A GENERATOR FOR STREAMING:
    We can't return the full response because it doesn't exist yet — the model
    is generating it token by token. A generator lets FastAPI pull tokens as
    they arrive and immediately forward them to the browser, creating the
    real-time typewriter effect. This is called Server-Sent Events (SSE) or
    HTTP streaming.
 
    The alternative (waiting for the full response) would mean:
      - User sees nothing for 5-15 seconds
      - Then the full answer appears all at once
    Streaming means:
      - First tokens appear in ~200ms
      - User sees the answer being written in real time
    Same total time, completely different perceived experience.

    PARAMETERS:
      query  : str       →  the user's question
      chunks : List[str] →  retrieved chunks from retriever.py
 
    YIELDS:
      str — one text token at a time (typically 1-4 words per yield)
    '''

    #Build the structured messages list
    messages = buildPrompt(query, chunks)

    #Call the Groq API with stream=True
    # client.chat.completions.create() is the standard OpenAI-compatible call.
    #
    # PARAMETERS:
    #
    # model=MODEL
    #   Which LLM to use. Defined as a constant above for easy swapping.
    #
    # messages=messages
    #   The assembled system + user messages from build_prompt().
    #
    # temperature=0.1
    #   Controls randomness in token selection. Range: 0.0 to 2.0.
    #   0.0 = fully deterministic (always picks highest-probability token)
    #   1.0 = default creative randomness
    #   0.1 = near-deterministic, slight variation allowed
    #   WHY 0.1 FOR LEGAL: We want factual, consistent answers — not creative
    #   paraphrasing. Low temperature keeps the model grounded to the context.
    #   High temperature would cause the model to "interpret" the legal text
    #   more liberally, which is dangerous.
    #
    # max_tokens=1024
    #   Maximum tokens in the response. 1024 ≈ 750 words — enough for a
    #   thorough answer to any single legal question. Prevents runaway responses.
    #   Groq's free tier has generous token limits but max_tokens is good
    #   practice to prevent accidental cost spikes if you ever switch to a
    #   paid API.
    #
    # stream=True
    #   Instead of waiting for the full response and returning it, the API
    #   returns an iterator that yields chunks as they're generated.
    #   Each chunk is a ChatCompletionChunk object (not a string).

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=.1,
        max_tokens=1024,
        stream=True
    )

    #Iterate over the stream
    # Each `chunk` from the completion stream is a ChatCompletionChunk object.
    # Its structure:
    #
    # ChatCompletionChunk(
    #   choices=[
    #     Choice(
    #       delta=ChoiceDelta(content="Hello", role=None),
    #       finish_reason=None,
    #     )
    #   ]
    # )
    #
    # chunk.choices[0].delta.content is the actual text token (a string).
    # On the final chunk, finish_reason will be "stop" and content will be None.
    # We check "if token" to skip those None values at the end of the stream.
    #
    # The "yield" keyword is what makes this a generator. Each token is handed
    # back to FastAPI immediately, which forwards it to the browser.

    for currChunk in completion:
        currToken = currChunk.choices[0].delta.content
        if currToken:
            yield currToken


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Non-streaming version (useful for testing)
# ─────────────────────────────────────────────────────────────────────────────

def getResponse(query: str, chunks:list[str]) ->str:
    '''
    Non-streaming version of the LLM call. Returns the complete response
    as a single string.
 
    WHY THIS EXISTS:
    Streaming is great for the UI but awkward to test in a Python script or
    in FastAPI's automatic docs (Swagger UI at /docs). This function lets you
    quickly test the full RAG pipeline from the terminal without needing a
    frontend:
 
        from ingest import ingest_pdf
        from retriever import retrieve_relevant_chunks
        from llm import get_response
 
        ingest_pdf("contract.pdf", "contract.pdf")
        chunks = retrieve_relevant_chunks("What is the termination clause?", "contract.pdf")
        print(get_response("What is the termination clause?", chunks))
 
    In production you'd use stream_response() for the actual API endpoint.
    '''

    messages = buildPrompt(query,chunks)

    completion = client.chat.completions.create(
        model = MODEL,
        messages=messages,
        temperature=.1,
        max_tokens=1024,
        stream=False
    )
    return completion.choices[0].message.content