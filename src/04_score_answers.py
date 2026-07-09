"""
Step 4: Use an LLM judge to mark each logged answer as correct/incorrect
against ground truth, and flag whether it's a hallucination (unsupported
by the retrieved context).

IMPORTANT: Before trusting this at scale, validate the judge against
~50 human-labeled examples (see validate_judge() below) to confirm it
agrees with human judgment. Report that agreement rate in your writeup --
it's a legitimate methods detail ("inter-rater reliability between the
LLM judge and human raters was X%").

Run: python src/04_score_answers.py --mode validate   (do this first)
     python src/04_score_answers.py --mode score      (then this)
"""
import argparse
import json

import pandas as pd

import config
from llm_client import call_llm

JUDGE_PROMPT = """You are grading an AI system's answer to a healthcare
question.

Question: {question}
Ground truth answer: {ground_truth}
Retrieved context given to the model: {context}
Model's answer: {model_answer}

Answer two questions in strict JSON format, nothing else:
1. "is_correct": true if the model's answer matches the ground truth in
   substance (not necessarily word-for-word), false otherwise.
2. "is_hallucination": true if the model's answer states something that
   is NOT supported by the retrieved context, false otherwise.

Respond with only: {{"is_correct": true/false, "is_hallucination": true/false}}"""


def judge_answer(question, ground_truth, context, model_answer, judge_model):
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth,
        context=context[:2000],  # truncate to keep judge prompt manageable
        model_answer=model_answer,
    )
    text = call_llm(judge_model, prompt).strip()
    try:
        parsed = json.loads(text)
        return parsed["is_correct"], parsed["is_hallucination"]
    except (json.JSONDecodeError, KeyError):
        print(f"WARNING: could not parse judge response: {text[:200]}")
        return None, None


def validate_judge(n=50):
    """
    Pulls n random rows, prints them so a human (you) can label them
    alongside the judge, so you can compute agreement rate before trusting
    the judge on the full dataset. Run this BEFORE scoring everything.
    """
    df = pd.read_csv(config.RESULTS_CSV)
    sample = df.sample(min(n, len(df)), random_state=42)
    sample.to_csv("results/judge_validation_sample.csv", index=False)
    print(f"Wrote {len(sample)} rows to results/judge_validation_sample.csv")
    print("Manually label an 'is_correct_human' column, then compare to "
          "the judge's output to compute agreement rate.")


def main():
    questions = pd.read_csv(config.QUESTIONS_CSV).set_index("question_id")
    results = pd.read_csv(config.RESULTS_CSV)

    is_correct_list = []
    is_hallucination_list = []

    for _, row in results.iterrows():
        q = questions.loc[row["question_id"]]
        # NOTE: context isn't saved in run_log.csv by default -- consider
        # adding a "context" column in 03_run_experiment.py if you want
        # the judge to check hallucination against the actual retrieved text.
        raw_context = row.get("context_sent_to_llm", "")
        context = "" if pd.isna(raw_context) else str(raw_context)

        is_correct, is_hallucination = judge_answer(
            question=q["question_text"],
            ground_truth=q["ground_truth_answer"],
            context=context,
            model_answer=row["model_answer"],
            judge_model=config.JUDGE_MODEL,
        )
        is_correct_list.append(is_correct)
        is_hallucination_list.append(is_hallucination)

    results["is_correct"] = is_correct_list
    results["is_hallucination"] = is_hallucination_list
    results.to_csv(config.RESULTS_CSV, index=False)
    print(f"Scored {len(results)} rows and updated {config.RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "score"], default="score")
    parser.add_argument("--n", type=int, default=15,
                         help="Number of rows to sample for --mode validate")
    args = parser.parse_args()

    if args.mode == "validate":
        validate_judge(n=args.n)
    else:
        main()