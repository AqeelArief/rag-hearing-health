"""
Step 1: Chunk raw documents in data/corpus/ into fixed-size token chunks
at each chunk size defined in config.py, and save the chunks with metadata.

Run: python src/01_build_corpus.py
"""
import os
import json
import pandas as pd
import tiktoken
from pypdf import PdfReader

import config

ENCODING = tiktoken.get_encoding("cl100k_base")


def load_document_text(filepath):
    """Reads a document. Supports .pdf (via pypdf) and plain .txt.
    Extend this further for .html, .docx, etc. if your corpus grows to
    include those formats."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(filepath)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts)
        if not text.strip():
            print(f"WARNING: extracted no text from {filepath} -- it may "
                  f"be a scanned/image-only PDF that needs OCR.")
        return text

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for {filepath}. "
            f"Add a handler in load_document_text() if you need this format."
        )


def chunk_text(text, chunk_size_tokens):
    """Simple fixed-size chunking by token count, no overlap.
    Consider adding ~10-15% overlap between chunks in a later iteration
    to avoid cutting sentences that matter."""
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size_tokens):
        chunk_tokens = tokens[i:i + chunk_size_tokens]
        chunks.append(ENCODING.decode(chunk_tokens))
    return chunks


def main():
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)
    metadata = pd.read_csv(config.CORPUS_METADATA_CSV)

    all_chunks = []
    for _, row in metadata.iterrows():
        filepath = os.path.join(config.CORPUS_DIR, row["filename"])
        if not os.path.exists(filepath):
            print(f"WARNING: missing file {filepath}, skipping.")
            continue

        text = load_document_text(filepath)

        for chunk_size in config.CHUNK_SIZES:
            chunks = chunk_text(text, chunk_size)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{row['doc_id']}_cs{chunk_size}_{idx}",
                    "doc_id": row["doc_id"],
                    "chunk_size": chunk_size,
                    "chunk_index": idx,
                    "text": chunk,
                    "source_org": row["source_org"],
                    "topic_category": row["topic_category"],
                })

    out_path = os.path.join(config.CHUNKS_DIR, "chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(all_chunks)} chunks across {len(config.CHUNK_SIZES)} "
          f"chunk sizes to {out_path}")


if __name__ == "__main__":
    main()