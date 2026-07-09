"""
Step 2: Embed chunks with each embedding model and build a separate
Chroma vector store per (embedding_model x chunk_size) combination.

Run: python src/02_embed_and_index.py
"""
import os
import json

import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_chunks():
    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.jsonl")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index(chunks, embedding_model_key, chunk_size):
    model_name = config.EMBEDDING_MODELS[embedding_model_key]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    subset = [c for c in chunks if c["chunk_size"] == chunk_size]
    if not subset:
        print(f"No chunks found for chunk_size={chunk_size}, skipping.")
        return

    texts = [c["text"] for c in subset]
    ids = [c["chunk_id"] for c in subset]
    metadatas = [
        {"doc_id": c["doc_id"], "topic_category": c["topic_category"]}
        for c in subset
    ]

    print(f"Embedding {len(texts)} chunks with {embedding_model_key} "
          f"(chunk_size={chunk_size})...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    collection_name = f"{embedding_model_key}_cs{chunk_size}"
    persist_dir = os.path.join(config.VECTOR_STORE_DIR, collection_name)
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    # Reset if it already exists, so re-runs don't duplicate data
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Saved collection '{collection_name}' with {len(texts)} vectors "
          f"to {persist_dir}")


def main():
    chunks = load_chunks()
    for embedding_model_key in config.EMBEDDING_MODELS:
        for chunk_size in config.CHUNK_SIZES:
            build_index(chunks, embedding_model_key, chunk_size)


if __name__ == "__main__":
    main()
