"""Kriteria 3 - versi CI: training model dalam MLflow Project, dijalankan oleh GitHub Actions.

Beda dari Membangun_model/modelling.py: tidak menunjuk tracking server manapun
(tidak ada set_tracking_uri) -> MLflow otomatis pakai file store lokal (mlruns/)
di runner, dan parameter diterima lewat argparse sesuai definisi entry point
di file MLProject.
"""

import argparse

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main(data_path, n_estimators, max_depth):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.autolog()
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="telco_preprocessing/telco_churn_clean.csv")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=20)
    a = p.parse_args()
    main(a.data_path, a.n_estimators, a.max_depth)
