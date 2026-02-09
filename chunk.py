# chunk.py
# Script to chunk the text into smaller chunks for the bot to process
import json


# Your chunk_text function
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# Load docs.json
with open("data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

all_chunks = []

for doc in docs:
    chunks = chunk_text(doc["text"], chunk_size=500)
    for chunk in chunks:
        all_chunks.append({
            "url": doc["url"],
            "text": chunk
        })

# Save chunks to a new JSON file
with open("data/docs_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_chunks)} chunks to data/docs_chunks.json")
