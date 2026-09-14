import os

SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"

os.environ["REQUESTS_CA_BUNDLE"] = SYSTEM_CA
os.environ["SSL_CERT_FILE"] = SYSTEM_CA
os.environ["CURL_CA_BUNDLE"] = SYSTEM_CA

print("Using certificate bundle:", SYSTEM_CA)

import requests
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from src.ingestion import ingest_documents


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# LOAD AND PREPARE DOCUMENTS
# ============================================================

print("Loading policy documents...")

chunks = ingest_documents()

if not chunks:
    raise RuntimeError("No policy chunks were created.")

print(f"Loaded {len(chunks)} policy chunks.")


# ============================================================
# LOAD BGE-LARGE EMBEDDING MODEL
# ============================================================

print("Loading BGE-large embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

texts = [
    chunk["text"]
    for chunk in chunks
]


# ============================================================
# GENERATE DOCUMENT EMBEDDINGS
# ============================================================

print("Generating document embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Embedding shape:", embeddings.shape)


# ============================================================
# BUILD FAISS VECTOR STORE
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print("FAISS index created.")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", dimension)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_policy(query, top_k=5):

    """
    Retrieve the most relevant policy chunks
    using BGE-large + FAISS.
    """

    # BGE v1.5 retrieval instruction
    query_text = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[idx]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    """
    Convert retrieved policy chunks into
    structured context for the LLM.
    """

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        source = metadata["source"]
        country = metadata["country"]
        policy_type = metadata["policy_type"]

        context += f"""
================ POLICY CHUNK {i} ================
Source: {source}
Country: {country}
Policy Type: {policy_type}

{result["text"]}

"""

    return context


# ============================================================
# CALL OLLAMA
# ============================================================

def ask_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["response"].strip()

    except Exception as e:

        return f"Ollama Error: {e}"


# ============================================================
# RAG ANSWER
# ============================================================

def rag_answer(query):

    """
    Complete RAG pipeline:

    User Question
        ↓
    BGE-large Embedding
        ↓
    FAISS Semantic Search
        ↓
    Top 5 Policy Chunks
        ↓
    Strict RAG Prompt
        ↓
    Ollama LLM
        ↓
    Answer + Sources
    """

    # --------------------------------------------------------
    # 1. Retrieve relevant policy chunks
    # --------------------------------------------------------

    results = search_policy(
        query,
        top_k=5
    )

    # --------------------------------------------------------
    # 2. Build policy context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # 3. Strict RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a company's travel and expense policy.

Your job is to answer the employee's question using ONLY the
POLICY CONTEXT provided below.

IMPORTANT RULES:

1. The POLICY CONTEXT is the source of truth.
2. Read ALL policy chunks before answering.
3. If ANY policy chunk contains information that answers the
   question, USE THAT INFORMATION.
4. Do not say "I could not find information" if the answer is
   present anywhere in the policy context.
5. Do not use outside knowledge.
6. Do not invent rules, limits, approvals, or exceptions.
7. If multiple chunks are relevant, combine them carefully.
8. If the question is about airport travel, use airport policy
   information when available.
9. If the question is about India, prefer India-specific policy.
10. If the question is about the US, prefer US-specific policy.
11. Always mention the source file used.

If the requested information genuinely does NOT exist in ANY
of the policy chunks, answer exactly:

"I could not find information about this in the available
policy documents."

Do not confuse "the information is in another chunk" with
"the information is unavailable."

Now answer the employee's question.

================ POLICY CONTEXT ================

{context}

================ EMPLOYEE QUESTION ================

{query}

================ ANSWER ================

Answer directly and concisely.
"""

    # --------------------------------------------------------
    # 4. Generate answer
    # --------------------------------------------------------

    answer = ask_ollama(
        prompt
    )

    # --------------------------------------------------------
    # 5. Collect unique sources
    # --------------------------------------------------------

    sources = []

    for result in results:

        source = result["metadata"]["source"]

        if source not in sources:
            sources.append(source)

    # --------------------------------------------------------
    # 6. Return complete RAG result
    # --------------------------------------------------------

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_results": results
    }


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    result = rag_answer(
        "What is the standard travel limit in India?"
    )

    print("\nQuestion:")
    print(result["query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print("-", source)

    print("\n")
    print("=" * 70)
    print("RAG TEST COMPLETED")
    print("=" * 70)
