import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources
nltk.download('punkt')


TEXT_DIR = "texts"
OUTPUT_PATH = "output/style_analysis.csv"

book_years = {
    "A little pretty pocket-Book.txt": 1744,
    "Goody two-shoes.txt": 1766,
    "The history of sandford and merton.txt": 1783,
    "The king of the golden river.txt": 1841,
    "Alice_s Adventuresin Wonderland.txt": 1866,
    "HEIDI.txt": 1880,
    "The Adventure of Pinocchio.txt": 1883,
    "Treasure island.txt": 1888,
    "THE WONDEFUL WIZRD OF OZ.txt": 1900,
    "Peter and Wendy.txt": 1911,
    "Charlie And The Chocolate Factory.txt": 1965,
    "matilda.txt": 1988,
    "Book 1 - The Philosopher_s Stone.txt": 1997,
    "coraline.txt": 2002,
    "Wonder.txt": 2012
}

conjunctions = {'and', 'but', 'because', 'so', 'or', 'yet'}

# Store results
results = []
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
# Loop through text files
for filename in os.listdir(TEXT_DIR):
    if not filename.endswith(".txt"):
        continue

    with open(os.path.join(TEXT_DIR, filename), 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into sentences
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Init counters
    total_words = 0
    total_word_length = 0
    short_sentences = 0
    long_sentences = 0
    questions = 0
    exclamations = 0
    statements = 0
    starts_with_conjunction = 0
    complex_sentences = 0
    
    # Analyze each sentence
    for sent in sentences:
        words = word_tokenize(sent)
        words_clean = [w.lower() for w in words if w.isalpha()]
        word_count = len(words_clean)
        total_words += word_count
        total_word_length += sum(len(w) for w in words_clean)

        if sent.strip().endswith('?'):
            questions += 1
        elif sent.strip().endswith('!'):
            exclamations += 1
        else:
            statements += 1

        if words_clean and words_clean[0] in conjunctions:
            starts_with_conjunction += 1

        if word_count <= 7:
            short_sentences += 1
        elif word_count >= 15:
            long_sentences += 1

        if any(len(w) > 10 for w in words_clean):
            complex_sentences += 1

    # Compute stats
    avg_sentence_length = round(total_words / total_sentences, 2) if total_sentences else 0
    avg_word_length = round(total_word_length / total_words, 2) if total_words else 0
    short_pct = round((short_sentences / total_sentences) * 100, 1) if total_sentences else 0
    long_pct = round((long_sentences / total_sentences) * 100, 1) if total_sentences else 0
    complex_pct = round((complex_sentences / total_sentences) * 100, 1) if total_sentences else 0
    starts_with_conj_pct = round((starts_with_conjunction / total_sentences) * 100, 1) if total_sentences else 0
    question_pct = round((questions / total_sentences) * 100, 1) if total_sentences else 0
    exclamation_pct = round((exclamations / total_sentences) * 100, 1) if total_sentences else 0

    #save results
    results.append({
        "book": filename,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "short_pct": short_pct,
        "long_pct": long_pct,
        "complex_pct": complex_pct,
        "questions": questions,
        "exclamations": exclamations,
        "starts_with_conjunction": starts_with_conjunction,
        "starts_with_conj_pct": starts_with_conj_pct,
        "question_pct": question_pct,
        "exclamation_pct": exclamation_pct,
        "total_sentences": total_sentences,
        "year": book_years.get(filename, None)

    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

