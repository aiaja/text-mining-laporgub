import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor



# Load Preprocessed Data
dataset = pd.read_csv('dataset/preprocessed_complaints.csv', sep=';')
X = dataset['complaints'].astype(str)
y = dataset['category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def define_models():
  """Defines base classifiers and meta classifier for stacking."""
  base_classifiers = [
      ('nb', MultinomialNB()),
      ('knn', KNeighborsClassifier(n_jobs=-1))
  ]
  meta_classifier = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
  return StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier, n_jobs=-1)

# Load TF-IDF
Tfidf_Vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, ngram_range=(2,2))

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', define_models()),
])

# Hyperparameter Tuning 
params = {
    'model__nb__alpha': [0.01, 0.1, 0.5, 1, 5],
    'model__knn__n_neighbors': [3, 5, 7, 9],
    'model__knn__weights': ['uniform', 'distance'],
    'model__knn__metric': ['euclidean', 'manhattan']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=10, cv=cv, scoring='accuracy', n_jobs=-1)
X_train = X_train.astype(str)

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