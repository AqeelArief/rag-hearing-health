# RAG Benchmarking for Hearing Healthcare Q&A

## Folder Structure

```
rag-hearing-health/
├── data/
│   ├── questions.csv          # your evaluation question set (you fill this in)
│   ├── corpus/                # raw source documents (PDFs/text you download)
│   └── corpus_metadata.csv    # metadata about each document
├── src/
│   ├── 01_build_corpus.py     # chunk documents, save chunk metadata
│   ├── 02_embed_and_index.py  # embed chunks, build vector stores per embedding model
│   ├── 03_run_experiment.py   # loop over all configs x questions, log results
│   ├── 04_score_answers.py    # LLM-as-judge scoring of correctness/hallucination
│   ├── 05_analyze_results.py  # statistical analysis (McNemar, logistic regression, bootstrap)
│   └── config.py              # central place to define your factorial grid
│   └──llm_client.py    
├── results/
│   └── run_log.csv            # the big long-format results table (auto-generated)
├── figures/                   # output plots
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama (for running open-source LLMs locally, free)
# Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com/download
ollama pull llama3.1
ollama pull qwen2.5
ollama pull mistral
ollama pull gemma2
```

## Workflow (run in this order)

1. **Fill in `data/questions.csv`** — this is manual work: write/collect 300–500 hearing-health Q&A pairs with ground truth (see schema in the file header).
2. **Drop source documents into `data/corpus/`** — download hearing health PDFs/pages from WHO, CDC, NIH, ASHA.
3. Run `python src/01_build_corpus.py` — chunks documents at 200/500/1000 tokens, saves to `data/chunks/`.
4. Run `python src/02_embed_and_index.py` — builds a Chroma vector store for each (embedding model × chunk size) combination.
5. Run `python src/03_run_experiment.py` — loops through every configuration in `config.py`, queries each question, logs results to `results/run_log.csv`.
6. Run `python src/04_score_answers.py` — uses an LLM judge to mark each answer correct/incorrect and flag hallucinations.
7. Open `notebooks/eda.ipynb` — explore your question set and corpus before diving into stats.
8. Run `python src/05_analyze_results.py` — runs the statistical tests and saves figures to `figures/`.

## Notes
- Start small: test the full pipeline end-to-end with ~20 questions and 2–3 configs before scaling to the full 500 × 24 grid. This catches bugs cheaply.
- Ollama models run locally and are free but slower than APIs — budget time accordingly for 12,000+ calls.
