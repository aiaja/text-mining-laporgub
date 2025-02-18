import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('dataset\datalaporgub-v2.csv')

# Data Cleaning
df = df.rename(columns={'content': 'complaints', 'sub_category_name': 'sub_categories', 'topic_name': 'topics'})
df = df.drop(['id', 'sub_categories', 'topics', 'category_id', 'sub_category_id', 'topic_id', 'created_at_date'], axis=1)
df = df.dropna(subset=['complaints'])
df = df.drop_duplicates(subset=['complaints', 'category'], keep='first')

# Text Preprocessing
df['complaints'] = df['complaints'].str.lower()
df['complaints'] = df['complaints'].apply(lambda x: re.sub(r'https?://\S+', '', x))
df['complaints'] = df['complaints'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['complaints'] = df['complaints'].apply(lambda x: re.sub(r'\d+', '', x))
df = df.drop_duplicates(subset=['complaints', 'category'], keep='first')

# Tokenization and Stopwords Removal
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
df['complaints'] = df['complaints'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
df['complaints'] = df['complaints'].apply(lambda x: [stemmer.stem(word) for word in x])

# Save Cleaned Dataset
df.to_csv('dataset\clean_dataset_part01.csv', sep=';')

# Plot Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['category'], kde=False)
plt.title('Distribution of Categories After Preprocessing')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()