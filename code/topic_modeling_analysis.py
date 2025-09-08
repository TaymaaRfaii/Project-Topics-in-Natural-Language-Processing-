# topic_modeling_analysis.py

"""
Topic Modeling Analysis for Children's Literature Dataset (1744–2012)

Steps:
1. Load and preprocess text files
2. Extract topics using LDA (Scikit-learn)
3. Analyze topic prevalence, overlap, and trends over time
4. Assign high-level story categories based on dominant topics
5. Output results as CSVs and graphs

Dependencies:
- pandas, numpy, matplotlib, seaborn, sklearn, nltk
"""

import os
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# === Setup ===
os.makedirs("output", exist_ok=True)

book_metadata = {
    'a little pretty pocket': ("A Little Pretty Pocket Book", 1744),
    'goody two-shoes': ("Goody Two-Shoes", 1766),
    'the history of sandf': ("The History of Sandford and Merton", 1783),
    'alice_s adventures': ("Alice in Wonderland", 1866),
    'the king of the gold': ("The King of the Golden River", 1841),
    'heidi': ("Heidi", 1880),
    'the adventure of pin': ("The Adventure of Pinocchio", 1883),
    'treasure island': ("Treasure Island", 1888),
    'the wondeful wiz': ("The Wonderful Wizard of Oz", 1900),
    'peter and wendy': ("Peter and Wendy", 1911),
    'charlie and the ch': ("Charlie and the Chocolate Factory", 1965),
    'matilda': ("Matilda", 1988),
    'book 1 - the philos': ("Harry Potter and the Philosopher’s Stone", 1997),
    'coraline': ("Coraline", 2002),
    'wonder': ("Wonder", 2012),
}

books = []
folder_path = "texts"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        path = os.path.join(folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        key = filename.lower()[:20]
        for pattern, (title, year) in book_metadata.items():
            if pattern in key:
                books.append({'title': title, 'year': year, 'text': content})
                break

books_df = pd.DataFrame(books)

# === Vectorize Texts ===
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(books_df['text'])

# === LDA Topic Modeling ===
lda = LatentDirichletAllocation(n_components=10, random_state=42)
topic_matrix = lda.fit_transform(X)
books_df['dominant_topic'] = topic_matrix.argmax(axis=1)

# === Theme Mapping ===
theme_map = {
    0: 'nature',
    1: 'school',
    2: 'identity',
    3: 'magic',
    4: 'adventure',
    5: 'family',
    6: 'morality',
    7: 'imagination',
    8: 'friendship',
    9: 'conflict',
}
books_df['theme'] = books_df['dominant_topic'].map(theme_map)

# === Topic Prevalence by Decade ===
books_df['decade'] = (books_df['year'] // 10) * 10
topic_by_decade = books_df.groupby('decade')['dominant_topic'].value_counts().unstack().fillna(0)
named_topic_by_decade = topic_by_decade.rename(columns=theme_map)
named_topic_by_decade.to_csv("output/topic_by_decade_named.csv")

# === Heatmap with Theme Names ===
normalized_named_topic_by_decade = named_topic_by_decade.div(named_topic_by_decade.sum(axis=1), axis=0)
plt.figure(figsize=(14, 8))
sns.heatmap(
    normalized_named_topic_by_decade.T,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5,
    cbar_kws={"label": "Proportion of Books"}
)
plt.title("Normalized Topic Prevalence by Decade", fontsize=16)
plt.xlabel("Decade")
plt.ylabel("Theme")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/topic_by_decade_heatmap_named.png", dpi=300)
plt.close()

# === Topic Overlap Between Books (Named & Rounded) ===
similarity_matrix = cosine_similarity(topic_matrix)
titles = books_df['title'].tolist()
similarity_df = pd.DataFrame(similarity_matrix, index=titles, columns=titles)
similarity_df.to_csv("output/book_similarity_named.csv")
similarity_df.round(2).to_csv("output/book_similarity_named_rounded.csv")

plt.figure(figsize=(14, 10))
sns.heatmap(
    similarity_df.round(2),
    xticklabels=titles,
    yticklabels=titles,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    square=True,
    cbar_kws={'label': 'Cosine Similarity'}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Topic Overlap Between Books", fontsize=16)
plt.tight_layout()
plt.savefig("output/book_similarity_heatmap_improved.png", dpi=300)
plt.close()

# === Top Words per Topic (with Theme Names) ===
terms = vectorizer.get_feature_names_out()
with open("output/topic_keywords.txt", "w", encoding="utf-8") as f:
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [terms[i] for i in topic.argsort()[:-11:-1]]
        label = theme_map.get(topic_idx, f"Topic {topic_idx}")
        f.write(f"Topic {topic_idx} ({label}): {' '.join(top_words)}\n")

# === Theme Distribution Over Time ===
theme_by_era = books_df.groupby('decade')['theme'].value_counts(normalize=True).unstack()
theme_by_era.to_csv("output/theme_by_era.csv")

theme_by_era.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
plt.title("Theme Distribution by Decade")
plt.ylabel("Proportion of Books")
plt.xlabel("Decade")
plt.tight_layout()
plt.savefig("output/theme_distribution_by_decade.png", dpi=300)
plt.close()

# === Assign Story Category Based on Theme ===
story_category_map = {
    'morality': 'Moral/Educational Stories',
    'family': 'Emotional/Realistic Stories',
    'school': 'Emotional/Realistic Stories',
    'identity': 'Emotional/Realistic Stories',
    'adventure': 'Adventure Stories',
    'magic': 'Fantasy/Fairy Tale Style',
    'imagination': 'Fantasy/Fairy Tale Style',
    'conflict': 'Fantasy/Fairy Tale Style',
    'friendship': 'Fantasy/Fairy Tale Style',
    'nature': 'Moral/Educational Stories',
}
books_df['story_category'] = books_df['theme'].map(story_category_map)
books_df[['title', 'year', 'dominant_topic', 'theme', 'story_category']].to_csv(
    "output/book_themes_and_categories.csv", index=False
)

# === Association Between Themes and Eras ===
theme_counts_by_era = books_df.groupby(['decade', 'theme']).size().unstack(fill_value=0)
theme_counts_by_era.to_csv("output/association_between_themes_and_eras.csv")

# === Heatmap: Association Between Themes and Eras ===
plt.figure(figsize=(12, 7))
sns.heatmap(
    theme_counts_by_era.T,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Book Count'}
)
plt.title("Association Between Themes and Eras", fontsize=16)
plt.xlabel("Decade")
plt.ylabel("Theme")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/association_between_themes_and_eras_heatmap.png", dpi=300)
plt.close()

# === Final Console Output ===
print("\n=== Books and Assigned Story Categories ===")
print(books_df[['title', 'theme', 'story_category']])
print("\n=== Association Between Themes and Eras ===")
print(theme_counts_by_era)
