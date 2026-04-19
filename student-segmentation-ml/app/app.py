import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.anime_recommender import AnimeRecommender

st.set_page_config(page_title="ML Assignment Hub", layout="wide")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose project:", ["Student Segmentation", "Anime Recommender"])

# =========================
# STUDENT SEGMENTATION
# =========================
if app_mode == "Student Segmentation":

    st.title("🎯 Student Segmentation")

    model_path = os.path.join(BASE_DIR, "models", "clustering_artifacts.pkl")

    # Auto train
    if not os.path.exists(model_path):
        st.warning("Training model... please wait ⏳")
        os.system("python -m src.train_model")

    if os.path.exists(model_path):

        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        model = artifacts["model"]
        scaler = artifacts["scaler"]
        features = artifacts["features"]

        age = st.number_input("Age", 10, 50, 18)
        friends = st.number_input("Friends", 0, 5000, 100)

        if st.button("Predict"):

            input_data = np.zeros((1, len(features)))

            for i, col in enumerate(features):
                if col == "age":
                    input_data[0, i] = age
                elif col == "numberoffriends":
                    input_data[0, i] = friends

            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)

            st.success(f"Cluster: {pred[0]}")

    else:
        st.error("Model creation failed.")

# =========================
# ANIME RECOMMENDER
# =========================
else:

    st.title("📺 Anime Recommender")

    @st.cache_resource
    def load():
        df = pd.read_csv(os.path.join(BASE_DIR, "data/anime.csv"))
        return AnimeRecommender(df)

    try:
        recommender = load()

        anime = st.selectbox("Select Anime", recommender.indices.index)

        if st.button("Recommend"):
            recs = recommender.recommend(anime)
            for r in recs:
                st.write(r)

    except:
        st.error("anime.csv missing")
