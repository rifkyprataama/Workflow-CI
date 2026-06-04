import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# [DIHAPUS/DINONAKTIFKAN] Biarkan GitHub Actions yang mengatur tracking URI-nya
# mlflow.set_tracking_uri("file:./mlruns") 

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
# MLflow akan otomatis menggunakan run yang dibuat oleh perintah `mlflow run` di YAML
with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Akurasi: {score:.3f}")

    # Mencatat model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # CARA PALING ANTI GAGAL: Simpan ID ke file .txt
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Selesai! Run ID berhasil disimpan ke run_id.txt.")