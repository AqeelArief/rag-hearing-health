"""
Step 3: For every configuration in config.py, run every question through
the retrieval + generation pipeline and log results.

Run: python src/03_run_experiment.py --grid pilot
     python src/03_run_experiment.py --grid full
"""
import argparse
import json
import os
import time

import chromadb
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from llm_client import call_llm

PROMPT_TEMPLATE = """You are a hearing healthcare assistant. Answer the
question using ONLY the context provided below. If the context does not
contain enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer concisely:"""

_embedding_model_cache = {}
_reranker_cache = None


def get_embedding_model(name):
    if name not in _embedding_model_cache:
        _embedding_model_cache[name] = SentenceTransformer(
            config.EMBEDDING_MODELS[name]
        )
    return _embedding_model_cache[name]


def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_cache


def dense_retrieve(question, embedding_model_key, chunk_size, top_k):
    collection_name = f"{embedding_model_key}_cs{chunk_size}"
    persist_dir = os.path.join(config.VECTOR_STORE_DIR, collection_name)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    model = get_embedding_model(embedding_model_key)
    query_emb = model.encode([question]).tolist()

    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    docs = results["documents"][0]
    ids = results["ids"][0]
    return list(zip(ids, docs))


def hybrid_retrieve(question, embedding_model_key, chunk_size, top_k):
    """Combine dense retrieval with BM25 keyword search, then merge by
    simple score fusion (rank-based)."""
    dense_results = dense_retrieve(question, embedding_model_key, chunk_size, top_k * 2)

    # Load all chunk texts for this chunk_size to build a BM25 index.
    # NOTE: for a real run, precompute/cache this once instead of rebuilding
    # per-question -- this is written for clarity, not speed.
    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.jsonl")
    all_chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_size"] == chunk_size:
                all_chunks.append(c)

    tokenized = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.split())
    bm25_ranked = sorted(
        zip([c["chunk_id"] for c in all_chunks], [c["text"] for c in all_chunks], scores),
        key=lambda x: x[2], reverse=True
    )[:top_k * 2]
    bm25_results = [(cid, text) for cid, text, _ in bm25_ranked]

    # Simple fusion: interleave dense and BM25 results, dedupe, cap at top_k
    combined = []
    seen = set()
    for pair in [item for pair in zip(dense_results, bm25_results) for item in pair]:
        if pair[0] not in seen:
            combined.append(pair)
            seen.add(pair[0])
    return combined[:top_k]


def rerank(question, retrieved, top_k):
    reranker = get_reranker()
    pairs = [[question, text] for _, text in retrieved]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:top_k]]


def query_llm(llm_key, prompt):
    # llm_key is one of config.LLMS (e.g. "llama3" or "gemini"), which
    # maps to an actual model name inside llm_client.MODEL_MAP.
    return call_llm(llm_key, prompt)


def run_single(question_row, cfg):
    question = question_row["question_text"]

    if cfg["retriever_type"] == "dense":
        retrieved = dense_retrieve(
            question, cfg["embedding_model"], cfg["chunk_size"], cfg["top_k"] * 2
        )
    else:
        retrieved = hybrid_retrieve(
            question, cfg["embedding_model"], cfg["chunk_size"], cfg["top_k"] * 2
        )

    if cfg["reranking"]:
        retrieved = rerank(question, retrieved, cfg["top_k"])
    else:
        retrieved = retrieved[:cfg["top_k"]]

    context = "\n\n".join([text for _, text in retrieved])

    # Cap context length sent to the LLM. This matters most for the
    # chunk_size=1000 config (5 retrieved chunks x up to 1000 tokens each
    # can exceed 5000 tokens of context alone), which combined with the
    # prompt template and requested output blows past Groq's free-tier
    # limit of 6000 tokens PER REQUEST (not just per minute). ~4000
    # characters is roughly 1000 tokens -- generous headroom while
    # still comparing configs meaningfully.
    MAX_CONTEXT_CHARS = 4000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    start = time.time()
    answer = query_llm(cfg["llm"], prompt)
    latency_ms = (time.time() - start) * 1000

    retrieved_doc_ids = [cid for cid, _ in retrieved]
    retrieval_correct = question_row["source_doc_id"] in "".join(retrieved_doc_ids)

    return {
        "question_id": question_row["question_id"],
        "config_id": cfg["config_id"],
        "chunk_size": cfg["chunk_size"],
        "embedding_model": cfg["embedding_model"],
        "retriever_type": cfg["retriever_type"],
        "reranking": cfg["reranking"],
        "llm": cfg["llm"],
        "top_k": cfg["top_k"],
        "retrieved_doc_ids": json.dumps(retrieved_doc_ids),
        "retrieval_correct": retrieval_correct,
        "context_sent_to_llm": context,
        "model_answer": answer.strip(),
        "latency_ms": latency_ms,
        # is_correct / is_hallucination filled in by 04_score_answers.py
    }


def load_existing_results():
    """
    If results/run_log.csv already exists from a previous (possibly
    interrupted) run, load it so we can skip (config_id, question_id) pairs
    that already succeeded -- this avoids burning API quota re-running
    questions you've already paid tokens for.
    """
    if os.path.exists(config.RESULTS_CSV):
        existing = pd.read_csv(config.RESULTS_CSV)
        done = set(zip(existing["config_id"], existing["question_id"]))
        print(f"Found existing results file with {len(existing)} rows "
              f"({len(done)} unique config/question pairs) -- resuming, "
              f"these will be skipped.")
        return existing, done
    return pd.DataFrame(), set()


def save_results(existing_df, new_results):
    """Merge newly collected results with whatever was already on disk and
    save. Called after every config finishes, so a crash/rate-limit/quota
    error partway through never loses previously completed work."""
    os.makedirs(os.path.dirname(config.RESULTS_CSV), exist_ok=True)
    combined = pd.concat([existing_df, pd.DataFrame(new_results)], ignore_index=True)
    combined.to_csv(config.RESULTS_CSV, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["pilot", "full"], default="pilot")
    parser.add_argument(
        "--llm", type=str, default=None,
        help="Comma-separated list of llm keys to include (e.g. 'gpt_oss_20b'). "
             "Useful for running only the models that still have API quota "
             "available, while others wait for a daily quota reset."
    )
    args = parser.parse_args()

    questions = pd.read_csv(config.QUESTIONS_CSV)
    grid = config.build_pilot_grid() if args.grid == "pilot" else config.build_full_grid()

    if args.llm:
        allowed_llms = {x.strip() for x in args.llm.split(",")}
        grid = [cfg for cfg in grid if cfg["llm"] in allowed_llms]
        print(f"Filtered to LLMs {allowed_llms}: {len(grid)} configs remain.")

    existing_df, done_pairs = load_existing_results()

    print(f"Running {len(grid)} configs x {len(questions)} questions "
          f"= {len(grid) * len(questions)} total calls "
          f"(skipping {len(done_pairs)} already completed)")

    for cfg in grid:
        print(f"\n--- Config: {cfg['config_id']} ---")
        n_questions = len(questions)
        config_results = []
        for i, (_, question_row) in enumerate(questions.iterrows(), start=1):
            pair = (cfg["config_id"], question_row["question_id"])
            if pair in done_pairs:
                print(f"  [{i}/{n_questions}] {question_row['question_id']} "
                      f"already done, skipping")
                continue

            q_start = time.time()
            try:
                result = run_single(question_row, cfg)
                config_results.append(result)
                elapsed = time.time() - q_start
                print(f"  [{i}/{n_questions}] {question_row['question_id']} "
                      f"done in {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{i}/{n_questions}] ERROR on "
                      f"{question_row['question_id']} / {cfg['config_id']}: {e}")

        # Save after every config, not just at the very end, so progress
        # survives a rate-limit/quota error or an interrupted run.
        if config_results:
            existing_df = save_results(existing_df, config_results)
            print(f"  Saved progress: {len(existing_df)} total rows in "
                  f"{config.RESULTS_CSV}")

    print(f"\nDone. {len(existing_df)} total rows in {config.RESULTS_CSV}")


if __name__ == "__main__":
    main()