import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import textstat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

TEXT_FOLDER = "textss"
book_years = {
    "A little pretty pocket-Book.txt": 1744,
    "Goody two-shoes.txt": 1766,
    "The history of sandford and merton.txt": 1783,
    "The king of the golden river.txt": 1841,
    "Alices Adventuresin Wonderland.txt": 1866,
    "HEIDI.txt": 1880,
    "The Adventure of Pinocchio.txt": 1883,
    "Treasure island.txt": 1888,
    "THE WONDEFUL WIZRD OF OZ.txt": 1900,
    "Peter and Wendy.txt": 1911,
    "Charlie And The Chocolate Factory.txt": 1965,
    "matilda.txt": 1988,
    "The Philosopher Stone.txt": 1997,
    "coraline.txt": 2002,
    "Wonder.txt": 2012
}

def get_era(year):
    if year < 1880:
        return 'early'
    elif year < 1965:
        return 'middle'
    else:
        return 'late'


def clean_text(text):
    start = text.find("*** STARTT")
    end = text.find("*** ENDD")
    if start != -1 and end != -1:
        return text[start+len("*** START")+1:end].lower().strip()
    else:
        return "Markers not found in the text."


stop_words = set(stopwords.words('english'))

def extract_features(text):
    flesch_score = textstat.flesch_reading_ease(text)
    dale_chall = textstat.dale_chall_readability_score(text)

    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words_clean = [w for w in words if w.isalpha()]
    pos_tags = nltk.pos_tag(words_clean)
    word_counts = Counter(words_clean)
    hapax_legomena = sum(1 for word, count in word_counts.items() if count == 1)
    hapax_ratio = hapax_legomena / len(words_clean) if words_clean else 0

    avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0
    avg_word_length = np.mean([len(w) for w in words_clean]) if words_clean else 0
    type_token_ratio = len(set(words_clean)) / len(words_clean) if words_clean else 0
    function_word_ratio = len([w for w in words_clean if w in stop_words]) / len(words_clean) if words_clean else 0
    complex_words = textstat.difficult_words(text) / len(words_clean) if words_clean else 0
    long_sentences = sum(1 for s in sentences if len(word_tokenize(s)) > 15)
    short_sentences = sum(1 for s in sentences if len(word_tokenize(s)) < 7)
    total_sentences = len(sentences)
    conjunctions = {'and', 'but', 'or', 'yet', 'so', 'because', 'although', 'though', 'while', 'whereas'}
    complex_sentences = sum(1 for s in sentences if sum(w in conjunctions for w in word_tokenize(s.lower())) > 1)


    return {
        'avg_sentence_length': avg_sentence_length,
        'hapax_legomena_ratio': hapax_ratio,
        'type_token_ratio': type_token_ratio,
        'function_word_ratio': function_word_ratio,
        'flesch_reading': flesch_score,
        'dale_chall': dale_chall,
        'complex_words': complex_words,
        'long_sentences_ratio': long_sentences / total_sentences if total_sentences else 0,
        'short_sentences_ratio': short_sentences / total_sentences if total_sentences else 0,
        'complex_sentences_ratio': complex_sentences / total_sentences if total_sentences else 0
    }


data = []
for filename, year in book_years.items():
    filepath = os.path.join(TEXT_FOLDER, filename)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    ctext = clean_text(text)
    era = get_era(year)
    features = extract_features(ctext)
    features['filename'] = filename
    features['era'] = era
    data.append(features)

df = pd.DataFrame(data)


print("Features extracted for each book:")
print(df[['filename', 'era'] + list(extract_features("").keys())])


X = df.drop(columns=['filename', 'era'])
y = df['era']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Linear SVM': SVC(kernel='linear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    print(f"\n=== {name} Results ===")

    model = make_pipeline(StandardScaler(), clf)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print("\nClassification Report (Stratified 5-Fold CV):")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=model.fit(X, y).named_steps[model.steps[-1][0]].classes_ if name == "Logistic Regression" else sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # Feature weights for Logistic Regression
    if name == "Linear SVM":
        model.fit(X, y)
        clf = model.named_steps['svc']
        print("\nFeature Weights:")
        for i, class_name in enumerate(clf.classes_):
            print(f"\nClass: {class_name}")
            for feat, coef in zip(X.columns, clf.coef_[i]):
                print(f"{feat}: {coef:.4f}")


from matplotlib import cm


X = df.drop(columns=['filename', 'era'])
eras = df['era']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

era_palette = {
    'early': '#1f77b4',
    'middle': '#ff7f0e',
    'late': '#2ca02c'
}


plt.figure(figsize=(14, 8))
sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    hue=eras,
    palette=era_palette,
    s=100,
    edgecolor='black'
)

plt.title("UMAP Clustering of Children's Books by Linguistic Features", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.legend(title="Era", loc="best")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()