"""
Central configuration for the RAG benchmarking experiment.

Edit this file to change your factorial grid. Start small (comment out
most options) to test the pipeline end-to-end before running the full sweep.

NOTE ON THIS VERSION:
- Embedding models switched to "small" variants (~130MB each instead of
  ~1.3GB) to fit on limited disk space.
- Reranking and top_k are now FIXED (not varied) to keep the grid at a
  clean 24 configurations: 3 chunk sizes x 2 embedding models x
  2 retriever types x 2 LLMs = 24.
- LLMs switched from local Ollama models to free-tier API models
  (Groq/Llama 3 and Google Gemini), called via llm_client.py, since
  Ollama requires macOS 12+ and this machine is on Catalina (10.15.7).
"""

import itertools

# --- Chunking ---
CHUNK_SIZES = [200, 500, 1000]  # tokens

# --- Embedding models (HuggingFace model names via sentence-transformers) ---
EMBEDDING_MODELS = {
    "bge": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/e5-small-v2",
}

# --- Retrieval strategy ---
RETRIEVER_TYPES = ["dense", "hybrid"]  # hybrid = dense + BM25

# --- Reranking (fixed on, not varied, to keep grid size manageable) ---
RERANKING = True
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Top-k documents retrieved (fixed, not varied) ---
TOP_K = 5

# --- LLMs ---
# Keys here must match the keys in llm_client.MODEL_MAP.
# Both models are served via Groq's free tier (see llm_client.py). Originally
# used Groq + Gemini for cross-provider comparison, but switched to two
# Groq models after llama-3.1-8b-instant was deprecated and Gemini's free
# daily quota proved too low (as little as 20 requests/day) for this
# project's account.
LLMS = ["gpt_oss_20b", "gpt_oss_120b"]

# --- Judge model for scoring correctness ---
# Also a key from llm_client.MODEL_MAP. Using the larger model here since
# it should be more reliable at following the strict JSON-output
# instructions in 04_score_answers.py.
JUDGE_MODEL = "gpt_oss_120b"

# --- Paths ---
QUESTIONS_CSV = "data/questions.csv"
CORPUS_DIR = "data/corpus"
CORPUS_METADATA_CSV = "data/corpus_metadata.csv"
CHUNKS_DIR = "data/chunks"
VECTOR_STORE_DIR = "data/vector_stores"
RESULTS_CSV = "results/run_log.csv"
FIGURES_DIR = "figures"


def build_full_grid():
    """
    Returns a list of config dicts representing every combination in the
    factorial design: 3 chunk sizes x 2 embedding models x 2 retriever
    types x 2 LLMs = 24 configurations. Reranking and top_k are fixed.
    """
    configs = []
    combos = itertools.product(
        CHUNK_SIZES,
        EMBEDDING_MODELS.keys(),
        RETRIEVER_TYPES,
        LLMS,
    )
    for chunk_size, emb_name, retriever, llm_name in combos:
        configs.append({
            "config_id": f"{emb_name}_{chunk_size}_{retriever}_{llm_name}",
            "chunk_size": chunk_size,
            "embedding_model": emb_name,
            "retriever_type": retriever,
            "reranking": RERANKING,
            "top_k": TOP_K,
            "llm": llm_name,
        })
    return configs


def build_pilot_grid():
    """
    A small starter grid (4 configs) spanning both LLMs, both retriever
    types, and two chunk sizes -- for testing the pipeline end-to-end
    before running the full 24-config sweep.
    """
    return [
        {"config_id": "pilot_bge_200_dense_20b", "chunk_size": 200,
         "embedding_model": "bge", "retriever_type": "dense",
         "reranking": RERANKING, "top_k": TOP_K, "llm": "gpt_oss_20b"},
        {"config_id": "pilot_bge_1000_dense_20b", "chunk_size": 1000,
         "embedding_model": "bge", "retriever_type": "dense",
         "reranking": RERANKING, "top_k": TOP_K, "llm": "gpt_oss_20b"},
        {"config_id": "pilot_e5_200_hybrid_120b", "chunk_size": 200,
         "embedding_model": "e5", "retriever_type": "hybrid",
         "reranking": RERANKING, "top_k": TOP_K, "llm": "gpt_oss_120b"},
        {"config_id": "pilot_bge_200_dense_120b", "chunk_size": 200,
         "embedding_model": "bge", "retriever_type": "dense",
         "reranking": RERANKING, "top_k": TOP_K, "llm": "gpt_oss_120b"},
    ]


if __name__ == "__main__":
    grid = build_full_grid()
    print(f"Full factorial grid has {len(grid)} configurations.")
    pilot = build_pilot_grid()
    print(f"Pilot grid has {len(pilot)} configurations.")