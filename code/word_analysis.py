import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textstat
from nltk.corpus import wordnet as wn
from nltk import ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import pandas as pd
import os

chart1 =[]
chart2 =[]
chart3=[]
chart4=[]
chart5=[]

outchart1 = "output/chart1.csv"
outchart2 = "output/chart2.csv"
outchart3 = "output/chart3.csv"
outchart4= "output/chart4.csv"
outchart5= "output/chart5.csv"
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

emotion_words = {
    'joy', 'happiness', 'happy', 'happily', 'joyful', 'sad', 'sadness', 'sadly', 'love', 'loving',
    'excited', 'excitement', 'enthusiasm', 'enthusiastic', 'content', 'contentment', 'pride', 'proud',
    'relief', 'grateful', 'gratitude', 'hope', 'hopeful', 'peaceful', 'peace', 'calm', 'relaxed',
    'satisfied', 'satisfaction', 'inspired', 'inspiration', 'affection', 'affectionate', 'amusement',
    'amused', 'bliss', 'cheerful', 'cheerfulness', 'delight', 'delighted', 'elation', 'ecstasy',
    'empathy', 'jubilation', 'optimism', 'optimistic', 'trust', 'trusting', 'fear', 'fearful',
    'scared', 'afraid', 'terrified', 'nervous', 'anxious', 'anxiety', 'uneasy', 'worried', 'worry',
    'hesitant', 'cautious', 'angry', 'anger', 'mad', 'furious', 'annoyed', 'annoying', 'irritated',
    'irritating', 'resentful', 'bitter', 'outraged', 'offended', 'frustrated', 'frustration', 'hurt',
    'miserable', 'pain', 'regret', 'regretful', 'remorse', 'remorseful', 'resentment', 'stress',
    'stressed', 'tense', 'tension', 'restless', 'drained', 'cry', 'cried', 'crying', 'lonely',
    'loneliness', 'tired', 'bored', 'depressed', 'depression', 'down', 'gloomy', 'despair',
    'hopeless', 'guilty', 'guilt', 'ashamed', 'shame', 'grief', 'grieving', 'sorrow', 'melancholy',
    'jealous', 'jealousy', 'confused', 'confusion', 'insecure', 'insecurity', 'overwhelmed',
    'pressured', 'passionate', 'attached', 'attachment', 'fond', 'fondness', 'cared', 'care', 'adoring',
    'adore', 'warming', 'heartbroken', 'mournful', 'bitter', 'regret', 'disappointed', 'disappointment',
    'grief', 'grieving', 'mourn', 'sobbing', 'sobs', 'tears', 'crying', 'brokenhearted', 'distressed',
    'distress', 'suffer', 'suffering', 'suffered', 'sufferer', 'pained', 'painful', 'agonized'
}

moral_words = {
    'good', 'evil', 'bad', 'virtuous', 'wicked', 'noble', 'immoral', 'benevolent', 'malevolent', 'righteous', 
    'sinful', 'honorable', 'corrupt', 'just', 'unjust', 'fair', 'unfair', 'right', 'wrong', 'pure', 'impure', 
    'innocent', 'guilty', 'ethical', 'unethical', 'rightful', 'unrightful', 'pious', 'vile', 'decent', 'undecent', 
    'moral', 'immoral', 'compassionate', 'cruel', 'generous', 'selfish', 'kind', 'unkind', 'charitable', 'greedy', 
    'respectful', 'disrespectful', 'honest', 'dishonest', 'loyal', 'disloyal', 'trustworthy', 'untrustworthy', 
    'caring', 'neglectful', 'altruistic', 'egotistical', 'good-hearted', 'bad-hearted', 'justifiable', 'unjustifiable', 
    'fair-minded', 'close-minded', 'humble', 'arrogant', 'selfless', 'self-centered', 'gracious', 'rude', 'honest', 
    'deceitful', 'generous', 'greedy', 'peaceful', 'violent', 'respectable', 'disreputable', 'honor', 'dishonor', 
    'holy', 'unholy', 'blameless', 'blameworthy', 'loving', 'hateful', 'gentle', 'harsh', 'merciful', 'cruel', 
    'helpful', 'unhelpful', 'compassionate', 'apathetic', 'forgiving', 'resentful', 'understanding', 'judgmental', 
    'righteousness', 'wickedness', 'gracious', 'mean', 'responsible', 'irresponsible', 'good-natured', 'bad-natured', 
    'pure-hearted', 'savage', 'moralistic', 'amoral', 'worthy', 'unworthy', 'unrepentant', 'repentant', 'innocence', 
    'guilt', 'self-discipline', 'recklessness', 'genuine', 'fake', 'honor-bound', 'dishonor-bound', 'fairness', 'bias', 
    'moral compass', 'unprincipled', 'socially responsible', 'irresponsible', 'accountable', 'unaccountable', 'virtuousness', 
    'immorality', 'temperance', 'intemperance', 'goodwill', 'badwill', 'solidarity', 'selfishness', 'equity', 'inequity', 
    'moral outrage', 'ethical dilemma', 'justice-oriented', 'evil-hearted', 'good-willed', 'unwilling', 'moral authority', 
    'moral relativism', 'moral clarity', 'educational integrity', 'intellectual honesty', 'civil', 'uncivil', 'wise', 'foolish', 
    'diligence', 'laziness', 'self-control', 'indulgence', 'fair play', 'unfair play', 'unethical', 'decent', 'indecent', 
    'principled', 'unprincipled', 'worthy', 'unworthy', 'honorific', 'dishonorific', 'meritorious', 'unmeritorious', 
    'appreciative', 'disrespectful', 'decent', 'malicious', 'charitable', 'vindictive', 'reputable', 'disreputable', 
    'goodwill', 'evil intent', 'respectable', 'contemptible', 'self-respecting', 'self-loathing', 'cultured', 'uncultured', 
    'humble', 'proud', 'industrious', 'lazy', 'loyal', 'betrayer', 'self-righteous', 'unrighteous', 'responsible', 'negligent', 
    'admirable', 'contemptible', 'fair-minded', 'biased', 'ethically sound', 'morally bankrupt', 'sacrifice', 'selfishness', 
    'patience', 'impatience', 'integrity', 'deceit', 'self-sacrifice', 'egoism', 'benevolence', 'selfishness', 'cooperation', 
    'self-serving', 'honorable', 'dishonorable', 'exemplary', 'unexemplary', 'moralistic', 'pragmatic', 'ethicality', 
    'unethicality', 'virtuosity', 'viciousness', 'noble-minded', 'low-minded', 'unpretentious', 'pretentious', 'altruism', 
    'self-interest', 'hardworking', 'lazy', 'generosity', 'meanness', 'sincerity', 'insincerity', 'temperate', 'intemperate', 
    'trustworthy', 'untrustworthy', 'humanitarian', 'cruel-hearted', 'sacrifice', 'self-preservation', 'goodwill', 'malice', 
    'respect', 'disdain', 'unselfish', 'selfish', 'fairness', 'partisanship', 'civility', 'barbarism', 'honesty', 'falsehood', 
    'helpful', 'obstructive', 'helpfulness', 'hinderance', 'tolerance', 'bigotry', 'acceptance', 'discrimination', 'charity', 
    'meanness', 'generosity', 'stinginess', 'care', 'neglect', 'mindfulness', 'carelessness', 'compassion', 'indifference'
}


def count_emotion_and_moral_words(tokens):
    emotion_count = sum(1 for word in tokens if word in emotion_words)
    moral_count = sum(1 for word in tokens if word in moral_words)
    return emotion_count, moral_count


def find_collocations(tokens, top_n=10):
    # Find bigrams (pairs of words)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    
    # Rank bigrams by pointwise mutual information (PMI)
    collocations = bigram_finder.nbest(BigramAssocMeasures.pmi, top_n)
    
    return collocations

def named_entity_recognition(tokens):
    # POS tagging
    pos_tags = pos_tag(tokens)
    
    # Named Entity Recognition (NER) chunking
    tree = ne_chunk(pos_tags)
    
    # Extract named entities
    named_entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):  # Subtree is a named entity
            entity = " ".join([word for word, tag in subtree])
            named_entities.append((entity, subtree.label()))  # entity and its type (e.g., PERSON, GPE)
    
    return named_entities


def get_lexical_field(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0].lexname()  # e.g., 'noun.plant', 'adj.all', etc.
    return None

def lexical_field_analysis(tokens):
    fields = []
    for word in tokens:
        lemma = lemmatizer.lemmatize(word)
        field = get_lexical_field(lemma)
        if field:
            fields.append(field)
    return Counter(fields)

def clean_text(text):
    # Find the index of the markers
    start = text.find("*** STARTT")
    end = text.find("*** ENDD")
    
    # If markers are found, return the cleaned content between them
    if start != -1 and end != -1:
        return text[start+len("*** START")+1:end].lower().strip()  # Clean and return content between markers
    else:
        return "Markers not found in the text."  # In case the markers are not found

def sentiment_analysis(text):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sid.polarity_scores(text)
    
    return sentiment_scores

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Initialize the Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def analyze_text(text, filename):
    # Flesch Reading Ease (higher is easier to read)
    flesch_score = textstat.flesch_reading_ease(text)

   # Dale-Chall Readability Score (lower is easier to read)
    dale_chall = textstat.dale_chall_readability_score(text)
    
    # Sentiment analysis
    sentiment_scores = sentiment_analysis(text)
     # Number of complex words (words with 3+ syllables)
    complex_words = textstat.difficult_words(text)

    # Preprocess the text
    tokens = preprocess_text(text)
    
    # Word frequency distribution
    freq_dist = FreqDist(tokens)
    
    # 1. Total word count and unique word count
    total_words = len(tokens)
    unique_words = len(set(tokens))
    
    # 2. Average word length
    avg_word_length = np.mean([len(word) for word in tokens])
    
    # 3. Top N most frequent words
    top_n = 5  # You can change N here
    top_words = freq_dist.most_common(3)
    
    # 4. POS Tagging
    pos_tags = pos_tag(tokens)
    
    emotion_count, moral_count = count_emotion_and_moral_words(tokens)
    remotion= emotion_count/total_words 
    rmoral= moral_count /total_words 

 
    # 5. Part-of-Speech Frequency (Nouns and Verbs)
    pos_counts = Counter(tag for word, tag in pos_tags)
    
    fields_counter = lexical_field_analysis(tokens)
    
    ner_results = named_entity_recognition(tokens)

    # Separate nouns and verbs
    nouns = [word for word, tag in pos_tags if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    verbs = [word for word, tag in pos_tags if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    adjectives = [word.lower() for word, tag in pos_tags if tag in ('JJ', 'JJR', 'JJS')]
    # Frequency distribution for Nouns and Verbs
    noun_freq = FreqDist(nouns)
    verb_freq = FreqDist(verbs)

    # Most common nouns and verbs
    top_nouns = noun_freq.most_common(3)
    top_verbs = verb_freq.most_common(3)
    adj_counter = Counter(adjectives)
    top_adjectives = adj_counter.most_common(5)
    positive = sentiment_scores['pos']
    neutral = sentiment_scores['neu']
    negative = sentiment_scores['neg']
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=fields_counter.most_common(10)
    rcomplex=complex_words/total_words
    runique=unique_words/ total_words
    
    #save results
    chart1.append({
        "book": filename,
        "Total_words": total_words,
        "ratio-unique_words": runique,
        "avg_word_length":  avg_word_length,
        "flesch_score": flesch_score,
        "dale_chall": dale_chall,
        "ratio of complex_words": rcomplex,
        "year": book_years.get(filename, None),
    })


    chart2.append({
        "book": filename,
        "3 most frequent words": top_words,
        "3 most frequent nouns": top_nouns,
        "3 most frequent verbs":  top_verbs,
        "3 most frequent adjectives": top_adjectives,
        "year": book_years.get(filename, None),
    })


    chart3.append({
        "book": filename,
        "a1":a1 ,
        "a2":a2,
        "a3":a3,
        "a4":a4,
        "a5":a5,
        "a6":a6 ,
        "a7":a7,
        "a8":a8,
        "a9":a9,
        "a10":a10,
        "year": book_years.get(filename, None),
    })
    chart4.append({
        "book": filename,
        "postive sent": positive,
        "netural sent": neutral,
        "negative sent":negative,
        "year": book_years.get(filename, None),
    })

    chart5.append({
        "book": filename,
        "ratio of words for emotion :": remotion,
        "ratio of words for morals :": runique,
        "year": book_years.get(filename, None),
    })

os.makedirs(os.path.dirname(outchart1), exist_ok=True)
os.makedirs(os.path.dirname(outchart2), exist_ok=True)
os.makedirs(os.path.dirname(outchart3), exist_ok=True)
os.makedirs(os.path.dirname(outchart4), exist_ok=True)
os.makedirs(os.path.dirname(outchart5), exist_ok=True)

# Loop through text files
for filename in os.listdir("textss"):
    if not filename.endswith(".txt"):
        continue

    with open(os.path.join("textss", filename), 'r', encoding='utf-8') as f:
        book_text = f.read()

    # Clean the text (extracts content between the markers)
    cleaned_text = clean_text(book_text)

    analyze_text(cleaned_text, filename)

df = pd.DataFrame(chart1)
df.to_csv(outchart1, index=False, encoding='utf-8-sig')
df = pd.DataFrame(chart2)
df.to_csv(outchart2, index=False, encoding='utf-8-sig')
df = pd.DataFrame(chart3)
df.to_csv(outchart3, index=False, encoding='utf-8-sig')
df = pd.DataFrame(chart4)
df.to_csv(outchart4, index=False, encoding='utf-8-sig')
df = pd.DataFrame(chart5)
df.to_csv(outchart5, index=False, encoding='utf-8-sig')

