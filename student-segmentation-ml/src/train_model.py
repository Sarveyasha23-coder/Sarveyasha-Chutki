import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def train():
    # Load dataset
    df = pd.read_csv("data/03_Clustering_Marketing.csv")

    df.columns = df.columns.str.lower().str.strip()

    # Basic cleaning
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['numberoffriends'] = pd.to_numeric(df['numberoffriends'], errors='coerce')

    df['age'].fillna(df['age'].median(), inplace=True)
    df['numberoffriends'].fillna(df['numberoffriends'].median(), inplace=True)

    # Select numeric features
    features = df.select_dtypes(include=np.number).columns
    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)

    # Save artifacts
    os.makedirs("models", exist_ok=True)

    with open("models/clustering_artifacts.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": list(features)
        }, f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train()
