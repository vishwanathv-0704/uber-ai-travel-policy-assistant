from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = BASE_DIR / "data" / "company_policy"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# 1. LOAD POLICY DOCUMENTS
# ============================================================

def load_documents():

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    documents = []

    for file_path in policy_files:

        try:

            content = file_path.read_text(
                encoding="utf-8"
            )

            if not content.strip():
                continue

            documents.append({
                "name": file_path.name,
                "content": content
            })

        except Exception as e:

            print(
                f"Error reading {file_path.name}: {e}"
            )

    return documents


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    text = "\n".join(lines).strip()

    return text


# ============================================================
# 3. METADATA
# ============================================================

def get_metadata(filename):

    if "travel_policy_india" in filename:

        return {
            "source": filename,
            "policy_type": "travel",
            "country": "India"
        }

    elif "travel_policy_us" in filename:

        return {
            "source": filename,
            "policy_type": "travel",
            "country": "US"
        }

    elif "airport_policy" in filename:

        return {
            "source": filename,
            "policy_type": "airport",
            "country": "Global"
        }

    elif "employee_eligibility" in filename:

        return {
            "source": filename,
            "policy_type": "eligibility",
            "country": "Global"
        }

    elif "expense_policy" in filename:

        return {
            "source": filename,
            "policy_type": "expense",
            "country": "Global"
        }

    elif "cancellation_policy" in filename:

        return {
            "source": filename,
            "policy_type": "cancellation",
            "country": "Global"
        }

    elif "approval_policy" in filename:

        return {
            "source": filename,
            "policy_type": "approval",
            "country": "Global"
        }

    else:

        return {
            "source": filename,
            "policy_type": "unknown",
            "country": "unknown"
        }


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        cleaned_content = clean_text(
            document["content"]
        )

        if not cleaned_content:
            continue

        metadata = get_metadata(
            document["name"]
        )

        split_texts = text_splitter.split_text(
            cleaned_content
        )

        for chunk_id, chunk_text in enumerate(
            split_texts
        ):

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_id
                }
            })

    return chunks


# ============================================================
# 5. COMPLETE INGESTION PIPELINE
# ============================================================

def ingest_documents():

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    return chunks


# ============================================================
# 6. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    documents = load_documents()

    print(
        f"Loaded {len(documents)} policy documents."
    )

    chunks = create_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\nSample chunk:\n")

    if chunks:

        print(
            "Metadata:",
            chunks[0]["metadata"]
        )

        print(
            "\nText:",
            chunks[0]["text"]
        )