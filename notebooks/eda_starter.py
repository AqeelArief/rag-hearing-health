"""
Starter EDA script -- convert to a Jupyter notebook (eda.ipynb) once you
have real data. Run cells to explore your question set and corpus before
moving to modeling/statistics.
"""
import pandas as pd
import matplotlib.pyplot as plt

questions = pd.read_csv("../data/questions.csv")

print(questions["topic_category"].value_counts())
print(questions["difficulty"].value_counts())
print(questions["answer_type"].value_counts())

questions["topic_category"].value_counts().plot(kind="bar")
plt.title("Question distribution by topic")
plt.tight_layout()
plt.show()

# Once you have a corpus_metadata.csv with real documents:
# corpus = pd.read_csv("../data/corpus_metadata.csv")
# print(corpus["source_org"].value_counts())
# print(corpus.groupby("source_org")["publish_year"].describe())
