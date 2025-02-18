import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load Preprocessed Data
dataset = pd.read_csv('dataset/clean_dataset_part01.csv', sep=';')
X = dataset['complaints'].astype(str)
y = dataset['category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.95, min_df=5)),
    ('model', MultinomialNB())
])

# Hyperparameter Tuning (Grid Search)
params = {'model__alpha': [0.01, 0.1, 0.5, 1, 5]}
model = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='accuracy')
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