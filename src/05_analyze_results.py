"""
Step 5: Statistical analysis of the results log.

Produces:
- Bootstrap confidence intervals on accuracy per configuration
- McNemar's test comparing pairs of configurations on the same questions
- Logistic regression: which factors (chunk_size, embedding_model, etc.)
  significantly predict is_correct
- Figures saved to config.FIGURES_DIR

Run: python src/05_analyze_results.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

import config


def bootstrap_ci_accuracy(df, config_id, n_boot=10000, seed=42):
    """95% bootstrap CI on accuracy for a single configuration."""
    subset = df[df["config_id"] == config_id]
    correct = subset["is_correct"].astype(int).values
    rng = np.random.default_rng(seed)
    boot_accs = []
    for _ in range(n_boot):
        sample = rng.choice(correct, size=len(correct), replace=True)
        boot_accs.append(sample.mean())
    lower, upper = np.percentile(boot_accs, [2.5, 97.5])
    return correct.mean(), lower, upper


def bootstrap_ci_difference(df, config_id_a, config_id_b, n_boot=10000, seed=42):
    """
    95% bootstrap CI on the accuracy DIFFERENCE between two configs,
    paired by question_id (only questions both configs answered).
    """
    a = df[df["config_id"] == config_id_a].set_index("question_id")["is_correct"].astype(int)
    b = df[df["config_id"] == config_id_b].set_index("question_id")["is_correct"].astype(int)
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values, b.loc[common].values

    rng = np.random.default_rng(seed)
    diffs = []
    n = len(common)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    observed_diff = a.mean() - b.mean()
    return observed_diff, lower, upper


def mcnemar_test(df, config_id_a, config_id_b):
    """
    Paired test for two classifiers on the same items.
    Returns (statistic, p_value).
    Uses the exact chi-square form: manually built 2x2 table of
    (both correct / A only / B only / both wrong).
    """
    a = df[df["config_id"] == config_id_a].set_index("question_id")["is_correct"].astype(bool)
    b = df[df["config_id"] == config_id_b].set_index("question_id")["is_correct"].astype(bool)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    b_only = int(((~a) & b).sum())   # A wrong, B correct
    a_only = int((a & (~b)).sum())   # A correct, B wrong

    n = a_only + b_only
    if n == 0:
        return None, None
    statistic = ((abs(a_only - b_only) - 1) ** 2) / n  # continuity correction
    p_value = 1 - chi2.cdf(statistic, df=1)
    return statistic, p_value


def logistic_regression_factors(df):
    """
    Model is_correct as a function of configuration factors to see which
    ones significantly predict accuracy, holding others constant.
    """
    model_df = df.copy()
    model_df["is_correct"] = model_df["is_correct"].astype(int)
    model_df["reranking"] = model_df["reranking"].astype(int)

    formula = ("is_correct ~ C(chunk_size) + C(embedding_model) + "
               "C(retriever_type) + reranking + C(top_k) + C(llm)")
    model = smf.logit(formula, data=model_df).fit(disp=False)
    print(model.summary())
    return model


def plot_accuracy_by_config(df, out_path):
    summary = df.groupby("config_id")["is_correct"].agg(["mean", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    plt.figure(figsize=(10, max(4, len(summary) * 0.3)))
    sns.barplot(data=summary, y="config_id", x="mean", color="steelblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Configuration")
    plt.title("Accuracy by RAG Configuration")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_heatmap_chunk_vs_embedding(df, out_path):
    pivot = df.pivot_table(
        index="chunk_size", columns="embedding_model",
        values="is_correct", aggfunc="mean"
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Accuracy: Chunk Size x Embedding Model")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_hallucination_by_topic(df, questions_df, out_path):
    merged = df.merge(questions_df[["question_id", "topic_category"]], on="question_id")
    summary = merged.groupby("topic_category")["is_hallucination"].mean().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    summary.plot(kind="bar", color="indianred")
    plt.ylabel("Hallucination rate")
    plt.title("Hallucination Rate by Topic Category")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(config.RESULTS_CSV)
    questions_df = pd.read_csv(config.QUESTIONS_CSV)

    # Drop rows where the LLM judge failed to produce a parseable verdict
    # (is_correct/is_hallucination came back as NaN). This can happen when
    # the judge model's output doesn't parse as strict JSON -- often on
    # long or unusually formatted model answers (e.g. markdown tables).
    # astype(int) below would otherwise crash on these NaN rows.
    n_before = len(df)
    unscored = df[df["is_correct"].isna()][["question_id", "config_id"]]
    df = df.dropna(subset=["is_correct"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"NOTE: dropped {n_dropped}/{n_before} rows "
              f"({n_dropped/n_before:.1%}) where the judge failed to "
              f"produce a parseable verdict:")
        for _, r in unscored.iterrows():
            print(f"  - {r['question_id']} / {r['config_id']}")
        print()

    # After dropping NaN rows, pandas may still leave these columns as
    # 'object' dtype (a holdover from mixing True/False/NaN). Cast
    # explicitly to bool so downstream aggregation (pivot_table, mean(),
    # seaborn plotting) gets a proper numeric/boolean type instead of
    # crashing on dtype=object.
    df["is_correct"] = df["is_correct"].astype(bool)
    df["is_hallucination"] = df["is_hallucination"].astype(bool)

    print("=== Bootstrap CIs per configuration ===")
    for cfg_id in df["config_id"].unique():
        acc, lo, hi = bootstrap_ci_accuracy(df, cfg_id)
        print(f"{cfg_id}: accuracy={acc:.3f} (95% CI: {lo:.3f}-{hi:.3f})")

    configs = df["config_id"].unique()
    if len(configs) >= 2:
        print("\n=== Example pairwise comparison (first two configs) ===")
        diff, lo, hi = bootstrap_ci_difference(df, configs[0], configs[1])
        print(f"Accuracy diff ({configs[0]} - {configs[1]}): "
              f"{diff:.3f} (95% CI: {lo:.3f}-{hi:.3f})")
        stat, p = mcnemar_test(df, configs[0], configs[1])
        if stat is not None:
            print(f"McNemar's test: statistic={stat:.3f}, p={p:.4f}")

    print("\n=== Logistic regression: which factors predict accuracy ===")
    try:
        logistic_regression_factors(df)
    except Exception as e:
        print(f"Logistic regression failed (likely too little data/variation yet): {e}")

    plot_accuracy_by_config(df, os.path.join(config.FIGURES_DIR, "accuracy_by_config.png"))
    plot_heatmap_chunk_vs_embedding(df, os.path.join(config.FIGURES_DIR, "heatmap_chunk_embedding.png"))
    plot_hallucination_by_topic(df, questions_df, os.path.join(config.FIGURES_DIR, "hallucination_by_topic.png"))


if __name__ == "__main__":
    main()