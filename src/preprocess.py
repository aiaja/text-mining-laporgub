import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv('dataset/new_data.csv')

# Data Cleaning: Rename columns and select relevant ones
df = df.rename(columns={'content': 'complaints', 'sub_category_name': 'sub_categories', 'topic_name': 'topics'})
df = df[['id', 'complaints', 'category']]
df['complaints'] = df['complaints'].astype(str)

# Remove empty complaints
df_hapus = df[~df['complaints'].str.contains(" ")]
df = df[~df.isin(df_hapus)].dropna()
df.set_index('id', inplace=True)

# Text Preprocessor (Ekphrasis)
text_processor = TextPreProcessor(
    normalize=['email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

# Preprocessing Functions
def clean_text(text):
    text = text_processor.pre_process_doc(text)
    text = ' '.join(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def remove_non_ascii(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def remove_emoji(text):
    return re.sub(r'([x#][A-Za-z0-9]+)', ' ', text)

def remove_special_chars(text):
    return re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single characters

def remove_url(text):
    return re.sub(r'http[s]?://\S+', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation.replace('_', '')))

def remove_rt(text):
    return text.replace('RT', " ").replace('hashtag','').replace('allcaps','').replace('elongated','').replace('repeated','').replace('emphasis','').replace('censored','')

def remove_mention(text):
    return re.sub(r'@[A-Za-z0-9_]+', '', text)

def lowercase(text):
    return text.lower()

def word_tokenize_wrapper(text):
    return word_tokenize(text)

# Apply preprocessing steps
df['complaints'] = df['complaints'].apply(lambda x: clean_text(x))
df['complaints'] = df['complaints'].apply(lambda x: remove_rt(x))
df['complaints'] = df['complaints'].apply(lambda x: remove_non_ascii(x))
df['complaints'] = df['complaints'].apply(lambda x: remove_url(x))
df['complaints'] = df['complaints'].apply(lambda x: remove_special_chars(x))
df['complaints'] = df['complaints'].apply(lambda x: remove_punctuation(x))
df['complaints'] = df['complaints'].apply(lambda x: lowercase(x))

# Tokenization
df['tokens'] = df['complaints'].apply(word_tokenize_wrapper)

# Load normalization dictionary
normalized_word = pd.read_excel("dataset/kamus_perbaikan_kata.xlsx")
normalized_word_dict = dict(zip(normalized_word.iloc[:, 0], normalized_word.iloc[:, 1]))

def normalized_term(document):
    return [normalized_word_dict.get(term, term) for term in document]

df['tokens_normalized'] = df['tokens'].apply(normalized_term)

# Stopwords Removal
def remove_stopwords(text):
    default_stopwords = set(stopwords.words('indonesian'))
    custom_stopwords = {'yang', 'di', 'itu', 'aku', 'kamu', 'saya', 'dia', 'ia', 'mereka', 'wr', 'wb', 'an', 'kak', 'dong', 'mas', 'br', 'wib', 'nah', 'ko', 'se', 'ber', 'al', 'ii', 'ny', 'ku', 'yah', 'kah', 'loh', 'lha', 'lho'}
    combined_stopwords = default_stopwords.union(custom_stopwords)
    return [word for word in text if word not in combined_stopwords]

df['tokens_no_stopwords'] = df['tokens_normalized'].apply(remove_stopwords)

# Stemming
stemmer = StemmerFactory().create_stemmer()

def apply_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(apply_stemming)

# Finalizing Data
df['category'] = df['category'].str.lower()
df = df[['stemmed_tokens', 'category']]
df = df.rename(columns={'stemmed_tokens': 'complaints'})

# Save preprocessed data
df.to_csv('dataset\preprocessed_complaints.csv', index=True, sep=';')
