import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load metadata
df = pd.read_csv("output/style_analysis.csv")
TEXT_DIR = "texts"

# Load text content
df["text"] = [
    open(os.path.join(TEXT_DIR, row["book"]), encoding="utf-8").read()
    if os.path.exists(os.path.join(TEXT_DIR, row["book"])) else ""
    for _, row in df.iterrows()
]

# Assign fixed periods
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

# Group texts by period
period_groups = df.groupby("period").agg({
    "text": lambda x: " ".join(x),
    "book": "count"
}).reset_index().sort_values(by="period")

# Build similarity table
rows = []
for i in range(len(period_groups) - 1):
    p1 = period_groups.iloc[i]
    p2 = period_groups.iloc[i + 1]

    try:
        tfidf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english').fit_transform([p1["text"], p2["text"]])
        cosine = round(cosine_similarity(tfidf)[0, 1], 3)
    except:
        cosine = 0.0

    action = "Combine" if cosine > 0.7 else "Keep"
    new_period = f"P{i+1}" if action == "Combine" else ""
    years = f"{p1['period']} and {p2['period']}"
    rows.append({
        "Segments": years,
        "Cosine similarity value": cosine,
        "Action": action,
        "New Period": new_period,
        "Years": years if action == "Combine" else "",
        "Number of Books": p1["book"] + p2["book"] if action == "Combine" else ""
    })

# Save output
out = pd.DataFrame(rows)
os.makedirs("output", exist_ok=True)
out.to_csv("output/final_period_table_by_fixed_periods.csv", index=False, encoding="utf-8-sig")
