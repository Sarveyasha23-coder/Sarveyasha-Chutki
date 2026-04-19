import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import sys
import json

# Fix path for src imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# =========================
# 🔐 KAGGLE AUTH (STREAMLIT SECRETS)
# =========================
try:
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_path):
        with open(kaggle_path, "w") as f:
            json.dump({
                "username": st.secrets["kaggle"]["username"],
                "key": st.secrets["kaggle"]["key"]
            }, f)
        os.chmod(kaggle_path, 0o600)

except Exception as e:
    st.warning("Kaggle secrets not found or not configured properly.")

# =========================
# 📦 IMPORT AFTER PATH FIX
# =========================
from src.anime_recommender import AnimeRecommender

# =========================
# 🎯 PAGE CONFIG
# =========================
st.set_page_config(page_title="ML Assignment Hub", layout="wide")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the project:", ["Student Segmentation", "Anime Recommender"])

# =========================
# 🎯 STUDENT SEGMENTATION
# =========================
if app_mode == "Student Segmentation":
    st.title("🎯 Student Segmentation (Clustering)")

    model_path = os.path.join(BASE_DIR, "models", "clustering_artifacts.pkl")

    # 🚀 Auto train if model not present
    if not os.path.exists(model_path):
        st.warning("Model not found. Training model... Please wait ⏳")
        try:
            os.system("python -m src.train_model")
        except:
            st.error("Training failed. Please check Kaggle setup.")
    
    # 🔁 Load model after training
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        model = artifacts["model"]
        scaler = artifacts["scaler"]
        transformer = artifacts["transformer"]
        features = artifacts["features"]

        st.write("### Predict Student Cluster")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 10, 50, 18)
            grad_year = st.number_input("Graduation Year", 2010, 2030, 2014)
        with col2:
            friends = st.number_input("Number of Friends", 0, 5000, 100)

        st.write("#### Interests")

        user_interests = {}
        demographic_cols = {'age', 'numberoffriends', 'gradyear'}
        interest_cols = [f for f in features if f.lower() not in demographic_cols]

        cols = st.columns(4)
        for i, feat in enumerate(interest_cols[:12]):
            user_interests[feat] = cols[i % 4].number_input(
                feat.capitalize(), 0, 10, 0
            )

        if st.button("Classify Student"):
            input_data = np.zeros((1, len(features)))
            feature_list = [f.lower() for f in features]

            # Fill basic features
            for col_name, val in [
                ('age', age),
                ('numberoffriends', friends),
                ('gradyear', grad_year)
            ]:
                if col_name in feature_list:
                    input_data[0, feature_list.index(col_name)] = val

            # Fill interests
            for feat, val in user_interests.items():
                if feat.lower() in feature_list:
                    input_data[0, feature_list.index(feat.lower())] = val

            input_df = pd.DataFrame(input_data, columns=features)
            input_transformed = transformer.transform(input_df)
            input_scaled = scaler.transform(input_transformed)

            prediction = model.predict(input_scaled)

            st.balloons()
            st.success(f"Cluster: **{prediction[0]}**")

    else:
        st.error("Model could not be created. Please check Kaggle API.")

# =========================
# 📺 ANIME RECOMMENDER
# =========================
elif app_mode == "Anime Recommender":
    st.title("📺 Anime Recommendation System")

    def download_anime_data():
        try:
            import kaggle
            st.info("Downloading anime dataset...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "CooperUnion/anime-recommendations-database",
                path="data",
                unzip=True
            )
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Download failed: {e}")

    anime_path = os.path.join(BASE_DIR, "data", "anime.csv")

    if not os.path.exists(anime_path):
        download_anime_data()

    if os.path.exists(anime_path):

        @st.cache_resource
        def load_recommender(path):
            df = pd.read_csv(path)
            return AnimeRecommender(df)

        recommender = load_recommender(anime_path)

        anime_list = sorted(list(recommender.indices.index))
        anime_name = st.selectbox("Select Anime", ["Select..."] + anime_list)

        if anime_name != "Select...":
            st.write(f"### Recommendations for {anime_name}")
            recs = recommender.recommend(anime_name)

            if isinstance(recs, str):
                st.warning(recs)
            else:
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")

    else:
        st.error("Anime dataset not found.")
