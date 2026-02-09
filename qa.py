# qa.py
# Script to answer questions using the FAISS index and the SQLite database according to the Overwolf documentation
import faiss
import sqlite3
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

FAISS_FILE = "data/index.faiss"
DB_FILE = "data/chunks.db"

# ---------------- LOAD FAISS INDEX ----------------
index = faiss.read_index(FAISS_FILE)


# ---------------- HELPERS ----------------
def search_chunks(question, k=10, max_distance=1.5):
    """
    Search for relevant chunks using FAISS.
    
    Args:
        question: The user's question
        k: Number of chunks to retrieve
        max_distance: Maximum L2 distance threshold (higher = more permissive)
                      Typical good chunks have distance < 1.0, very relevant < 0.8
    """
    # Create a NEW DB connection per call (thread-safe)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    q_vector = np.array([q_embedding], dtype=np.float32)
    # Search for more chunks initially, then filter by distance
    distances, indices = index.search(q_vector, k * 2)
    
    print(f"FAISS search returned indices: {indices[0][:k]}")
    print(f"FAISS distances: {distances[0][:k]}")

    # Get all chunk IDs in order to map FAISS indices correctly
    # This ensures mapping works even if FAISS was built without ORDER BY
    cursor.execute("Select id FROM chunks ORDER BY id")
    all_ids = [row[0] for row in cursor.fetchall()]
    
    # Verify FAISS and DB counts match
    if index.ntotal != len(all_ids):
        print(f"[WARNING] FAISS index has {index.ntotal} vectors but DB has {len(all_ids)} rows!")
    
    chunks = []
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        
        # Filter out chunks that are too distant (not relevant)
        if distance > max_distance:
            print(f"Skipping chunk at idx {idx} (distance {distance:.3f} > {max_distance})")
            continue
            
        if idx < len(all_ids):
            db_id = all_ids[idx]
            cursor.execute(
                "Select text, url FROM chunks WHERE id = ?",
                (db_id,)
            )
            row = cursor.fetchone()
            if row:
                print(f"FAISS idx {idx} -> DB ID {db_id}: Found chunk ({len(row[0])} chars, distance: {distance:.3f})")
                chunks.append({"text": row[0], "url": row[1]})
            else:
                print(f"[WARNING] FAISS idx {idx} -> DB ID {db_id}: No row found!")
        else:
            print(f"[WARNING] FAISS idx {idx} is out of range (max: {len(all_ids)-1})")
        
        # Stop once we have enough chunks
        if len(chunks) >= k:
            break
    
    print(f"Retrieved {len(chunks)} chunks from database (after distance filtering)")
    if len(chunks) == 0:
        print("No chunks retrieved! This will cause 'I couldn't find the answer'")

    conn.close()
    return chunks

# ---------------- QA ----------------
def answer_question(question):
    print(f"Answering question: '{question[:100]}...'")
    chunks = search_chunks(question, k=10)  # Increased from 5 to 10 for better coverage

    if not chunks:
        print("[ERROR] No chunks found - context will be empty")
        context = ""
    else:
        context = "\n\n".join(
            f"Source: {c['url']}\n{c['text']}"
            for c in chunks
        )
        print(f"Built context with {len(chunks)} chunks, {len(context)} characters")
        # Show preview of first chunk to verify relevance
        if chunks:
            first_chunk_preview = chunks[0]['text'][:300] + "..." if len(chunks[0]['text']) > 300 else chunks[0]['text']
            print(f"First chunk preview: {first_chunk_preview}")
            print(f"First chunk URL: {chunks[0]['url']}")

# ---------------- RESPONSE PROMPT---------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly and helpful Overwolf documentation assistant. "
                    "Your goal is to answer questions using the provided documentation. "
                    "\n\n"
                    "IMPORTANT INSTRUCTIONS:\n"
                    "- Always try to provide a helpful answer based on the documentation, even if it's not a perfect match.\n"
                    "- If the documentation contains ANY relevant information (even partial), use it to construct an answer.\n"
                    "- You can infer reasonable answers from related documentation sections.\n"
                    "- Only say 'I couldn't find the answer in the documentation.' as a LAST RESORT when the documentation is completely unrelated to the question.\n"
                    "- If multiple documentation sections are provided, synthesize information from all relevant parts.\n"
                    "- Be concise but thorough in your answers."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Relevant Documentation:\n{context}\n\n"
                    f"Please answer the question using the documentation above. If the documentation contains any relevant information, use it to provide a helpful answer."
                )
            }
        ],
    )

    return response.choices[0].message.content
# ---------------- TEST ----------------
# Uncomment this to test the QA system locally
#if __name__ == "__main__":
#    while True:
#        q = input("Ask Overwolf docs: ")
#        print("\n" + answer_question(q) + "\n")
