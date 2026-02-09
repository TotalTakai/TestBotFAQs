# embed.py
import os
import sqlite3
import numpy as np
import faiss
from openai import OpenAI
import time
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- CONFIG ----------------
CHUNK_FILE = "data/docs_chunks.json"      # raw chunks from crawler
DB_FILE = "data/chunks.db"                # SQLite DB to store embeddings
FAISS_FILE = "data/index.faiss"
BATCH_SIZE = 500
EMBED_MODEL = "text-embedding-3-small"
DIMENSION = 1536
MAX_TOKENS = 8000   # max tokens per chunk for embeddings
WORDS_PER_TOKEN = 0.75  # approx conversion

# ---------------- INIT ----------------
# Validate API key
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

client = OpenAI(API_KEY)

# Load chunks
if not os.path.exists(CHUNK_FILE):
    raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}. Please run chunk.py first to generate it.")

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks from {CHUNK_FILE}")

# Setup SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    text TEXT,
    embedding BLOB
)
""")
conn.commit()

# Load existing embeddings from DB into FAISS
# IMPORTANT: Use ORDER BY id to ensure FAISS index positions match database IDs
cursor.execute("SELECT embedding FROM chunks ORDER BY id")
rows = cursor.fetchall()
index = faiss.IndexFlatL2(DIMENSION)
if rows:
    vectors = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    index.add(vectors)
    print(f"Loaded {len(vectors)} embeddings into FAISS (ordered by ID)")

# Determine start index (resume from where we left off)
cursor.execute("SELECT COUNT(*) FROM chunks")
start_index = cursor.fetchone()[0]
print(f"Resuming from chunk {start_index} (out of {len(chunks)} total chunks)")

# ---------------- HELPERS ----------------
def save_embedding_to_db(url, text, vector):
    cursor.execute(
        "INSERT INTO chunks (url, text, embedding) VALUES (?, ?, ?)",
        (url, text, vector.tobytes())
    )
    conn.commit()

def embed_chunk_recursive(text, max_tokens=MAX_TOKENS):
    """
    Recursively split text until each piece fits the model's token limit.
    Returns list of (text, vector) tuples.
    """
    words = text.split()
    words_per_chunk = int(max_tokens * WORDS_PER_TOKEN)

    # If the chunk is small enough, embed it
    if len(words) <= words_per_chunk:
        while True:
            try:
                emb = client.embeddings.create(model=EMBED_MODEL, input=text)
                vector = np.array(emb.data[0].embedding, dtype=np.float32)
                return [(text, vector)]
            except Exception as e:
                # Only retry network/API errors, not max token errors
                if "maximum context length" in str(e):
                    # Split further if API still rejects (rare)
                    mid = len(words) // 2
                    first_half = " ".join(words[:mid])
                    second_half = " ".join(words[mid:])
                    return embed_chunk_recursive(first_half, max_tokens) + embed_chunk_recursive(second_half, max_tokens)
                else:
                    print(f"Error embedding chunk: {e}, retrying in 5s")
                    time.sleep(5)
    else:
        # Chunk too big → split before calling API
        mid = len(words) // 2
        first_half = " ".join(words[:mid])
        second_half = " ".join(words[mid:])
        return embed_chunk_recursive(first_half, max_tokens) + embed_chunk_recursive(second_half, max_tokens)


# ---------------- MAIN LOOP ----------------
for i in range(start_index, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    for j, chunk in enumerate(batch):
        embedded_vectors = embed_chunk_recursive(chunk["text"])
        for sub_text, vector in embedded_vectors:
            save_embedding_to_db(chunk["url"], sub_text, vector)
            index.add(np.array([vector]))
        print(f"Processed chunk {i+j+1}/{len(chunks)} with {len(embedded_vectors)} sub-chunks")

    # Save FAISS index after each batch
    faiss.write_index(index, FAISS_FILE)
    print(f"Processed batch {i} -> {i+len(batch)}, FAISS index saved at {FAISS_FILE}")
    time.sleep(1)  # gentle rate limit

print("Embedding complete!")
conn.close()
