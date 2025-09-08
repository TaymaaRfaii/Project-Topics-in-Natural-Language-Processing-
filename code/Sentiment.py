import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("output/chart4.csv")
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
df = df.rename(columns={
    "postive_sent": "positive_sent",
    "netural_sent": "neutral_sent"
})


required_cols = {"year", "positive_sent", "neutral_sent", "negative_sent"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")


df = df.dropna(subset=["year", "positive_sent", "neutral_sent", "negative_sent"])

# Classify periods
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

# Group and average sentiment by period
grouped = df.groupby("period", observed=True).agg({
    "positive_sent": "mean",
    "neutral_sent": "mean",
    "negative_sent": "mean"
}).reset_index()

# Plot the sentiment trends
plt.figure(figsize=(10, 6))
sns.lineplot(x="period", y="positive_sent", data=grouped, label="Positive Sentiment", marker="o", color="green")
sns.lineplot(x="period", y="neutral_sent", data=grouped, label="Neutral Sentiment", marker="o", color="gray")
sns.lineplot(x="period", y="negative_sent", data=grouped, label="Negative Sentiment", marker="o", color="red")

plt.title("Sentiment Trends by Period", fontsize=14)
plt.xlabel("Period", fontsize=12)
plt.ylabel("Average Sentiment Score", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

os.makedirs("charts_by_fourth_file", exist_ok=True)
plt.savefig("charts_by_fourth_file/sentiment_by_period.png")
plt.show()
