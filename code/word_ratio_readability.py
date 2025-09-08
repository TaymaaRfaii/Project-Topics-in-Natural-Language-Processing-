import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("output/chart1.csv")

def classify_period(year):
    if pd.isna(year):
        return "Unknown"
    year = int(year)
    if 1701 <= year <= 1800:
        return "1700–1799"
    elif 1801 <= year <= 1900:
        return "1800–1899"
    elif 1901 <= year <= 2000:
        return "1900–1999"
    elif year > 2000:
        return "2000+"
    else:
        return "Unknown"

df["period"] = df["year"].apply(classify_period)

grouped = df.groupby("period").agg({
    "book": "count",
    "Total_words": "sum",
    "ratio-unique_words": "mean",
    "avg_word_length": "mean",
    "flesch_score": "mean",
    "dale_chall": "mean",
    "ratio of complex_words": "mean"
}).reset_index()



grouped.rename(columns={"book": "num_books"}, inplace=True)
grouped = grouped.round(2)


label_map = {
    "ratio-unique_words": "Unique Words / Total Words Ratio",
    "avg_word_length": "Average Word Length",
    "flesch_score": "Flesch Reading Score",
    "dale_chall": "Dale-Chall Readability Score",
    "ratio of complex_words": "Average Complex Words"
}

period_order = ["1700–1799", "1800–1899", "1900–1999", "2000+"]


grouped["period"] = pd.Categorical(grouped["period"], categories=period_order, ordered=True)
grouped = grouped.sort_values("period")


for metric in label_map:
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="period", y=metric, data=grouped, marker="o")
    plt.title(f"{label_map[metric]} by Period")
    plt.xlabel("Period")
    plt.ylabel(label_map[metric])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"charts_by_second_file/{metric.replace(' ', '_')}_by_period.png")
    plt.close()
