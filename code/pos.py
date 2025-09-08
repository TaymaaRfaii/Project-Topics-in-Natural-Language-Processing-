import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("output/chart33.csv")

for col in ['a1', 'a2', 'a3', 'a4']:
    df[col] = df[col].str.extract(r"(\d+)").astype(float)

# Convert Total_words to float (if not already)
df['Total_words'] = pd.to_numeric(df['Total_words'], errors='coerce')


for col in ['a1', 'a2', 'a3', 'a4']:
    df[f"{col}_pct"] = (df[col] / df['Total_words']) * 100

# Classify years into historical periods
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
    "a1_pct": "mean",
    "a2_pct": "mean",
    "a3_pct": "mean",
    "a4_pct": "mean"
}).reset_index()


grouped = grouped.round(2)


period_order = ["1700–1799", "1800–1899", "1900–1999", "2000+"]
grouped["period"] = pd.Categorical(grouped["period"], categories=period_order, ordered=True)
grouped = grouped.sort_values("period")

label_map = {
    "a1_pct": "Noun: Person (%)",
    "a2_pct": "Noun: Artifact (%)",
    "a3_pct": "Adjective (%)",
    "a4_pct": "Adverb (%)"
}

os.makedirs("charts_by_category", exist_ok=True)

for col in label_map:
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="period", y=col, data=grouped, marker="o")
    plt.title(f"Average {label_map[col]} per Book by Period")
    plt.xlabel("Period")
    plt.ylabel(label_map[col])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"charts_by_category/{col}_by_period.png")
    plt.close()
