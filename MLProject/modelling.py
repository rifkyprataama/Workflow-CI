import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

print("Memuat dataset...")
df = pd.read_csv("namadataset_preprocessing/dataset_clean.csv")
df = df.dropna(subset=['ulasan_bersih']).reset_index(drop=True)

X = df['ulasan_bersih']
y = df['sentimen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Membangun Pipeline (TF-IDF + Random Forest)...")
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    RandomForestClassifier(n_estimators=50, random_state=42)
)

print("Melatih model...")
with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Akurasi: {score:.3f}")

    # Mencatat model
    mlflow.sklearn.log_model(pipeline, "model")

print("Selesai!")