
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load data
df = pd.read_csv("output/style_analysis.csv")

# Classify years into periods
def classify_period(year):
    if 1700 <= year <= 1799:
        return "1700–1799"
    elif 1800 <= year <= 1899:
        return "1800–1899"
    elif 1900 <= year <= 1999:
        return "1900–1999"
    elif year >= 2000:
        return "2000+"
    else:
        return "Unknown"

df["period"] = df["year"].apply(classify_period)

# Calculate the number of books and average statistical values (sentence length, percentages, etc.)
# for each time period, along with total counts of questions and exclamations
grouped = df.groupby("period").agg({
    "book": "count",
    "avg_sentence_length": "mean",
    "short_pct": "mean",
    "long_pct": "mean",
    "complex_pct": "mean",
    "questions": "sum",
    "exclamations": "sum"
}).reset_index()

grouped.rename(columns={"book": "num_books"}, inplace=True)
grouped = grouped.round(2)

# saving results
grouped.to_csv("output/sentence_analysis_by_period.csv", index=False)

label_map = {
    "avg_sentence_length": "Average Sentence Length (words)",
    "short_pct": "Short Sentences (%) ≤ 7 words",
    "long_pct": "Long Sentences (%) ≥ 15 words",
    "complex_pct": "Complex Sentences (%) with long words",
    "questions": "Total Question Sentences",
    "exclamations": "Total Exclamatory Sentences"
}

period_order = ["1700–1799", "1800–1899", "1900–1999", "2000+"]

# Create a folder to save the charts
os.makedirs("charts_by_period", exist_ok=True)

for metric in label_map:
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="period", y=metric, data=grouped, marker="o", sort=False)
    plt.title(f"{label_map[metric]} by Period")
    plt.xlabel("Period")
    plt.ylabel(label_map[metric])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"charts_by_period/{metric}_by_period.png")
    

