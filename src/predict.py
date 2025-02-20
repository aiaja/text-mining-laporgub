import pandas as pd
import joblib
import os

# Load Dataset Hasil Preprocessing
dataset = pd.read_csv('dataset/clean_dataset_part01.csv', sep=';')
X = dataset['complaints'].astype(str)

# Load Model Terlatih
model_path = 'models/naive_bayes_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di {model_path}. Jalankan train.py terlebih dahulu.")
model = joblib.load(model_path)

# Lakukan Prediksi
predictions = model.predict(X)
dataset['predicted_category'] = predictions

def predict_category(model, text):
    """Melakukan prediksi kategori dari teks."""
    return model.predict([text])[0]

# Ekspor fungsi agar dapat diimpor
__all__ = ['predict_category']

# Simpan Hasil Prediksi ke CSV (Best Practice: Simpan hasil untuk audit dan reanalisis)
output_path = 'dataset/predictions.csv'
dataset.to_csv(output_path, sep=';', index=False)
print(f"Predictions saved to {output_path}")