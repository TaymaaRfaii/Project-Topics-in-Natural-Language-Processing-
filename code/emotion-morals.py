import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("output/chart5.csv")
df = df.dropna(subset=["year", "ratio of words for emotion :", "ratio of words for morals :"])

def classify_period(year):
    year = float(year)
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

period_order = ["1700–1799", "1800–1899", "1900–1999", "2000+"]
df["period"] = pd.Categorical(df["period"], categories=period_order, ordered=True)

grouped = df.groupby("period").agg({
    "ratio of words for emotion :": "mean",
    "ratio of words for morals :": "mean"
}).reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x="period", y="ratio of words for emotion :", data=grouped, label="Emotion Words", marker="o")
sns.lineplot(x="period", y="ratio of words for morals :", data=grouped, label="Moral Words", marker="o")

os.makedirs("charts_by_fifth_file", exist_ok=True)
plt.title("Emotion vs Moral Word Counts by Period", fontsize=14)
plt.xlabel("Period", fontsize=12)
plt.ylabel("Average Word Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("charts_by_fifth_file/emotion_vs_moral_by_period.png")
plt.show()
