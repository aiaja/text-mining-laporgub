import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load Preprocessed Data
dataset = pd.read_csv('dataset/clean_dataset_part01.csv', sep=';')
X = dataset['complaints'].astype(str)
y = dataset['category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Load TF-IDF
Tfidf_Vectorizer = TfidfVectorizer()
X_train_vec = Tfidf_Vectorizer.fit_transform(X_train)

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultinomialNB())
])

# Hyperparameter Tuning (Grid Search)
params = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 3],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'model__alpha': [0.1, 0.5, 1.0]
}
model = GridSearchCV(pipeline, params, cv = 5, scoring='f1_weighted', verbose=2, n_jobs=-1)
model.fit(X_train, y_train)

# Simpan Model ke Folder 'models'
os.makedirs('models', exist_ok=True)
model_path = 'models/naive_bayes_model.pkl'
joblib.dump(model.best_estimator_, model_path)
print(f"Model saved successfully at {model_path}")

# Informasi Model Terbaik
print(f"Best Parameters: {model.best_params_}")
print(f"Training Accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.2f}")