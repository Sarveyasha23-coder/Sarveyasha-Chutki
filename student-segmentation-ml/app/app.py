import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import sys

# Ensure project root is in path so 'src' can be imported when running from any directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anime_recommender import AnimeRecommender

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Map custom environment variables to standard Kaggle keys for deployment environments
if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_TOKEN" in os.environ:
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]
if "KAGGLE_USERNAME" not in os.environ and "KAGGLE_USER" in os.environ:
    os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USER"]

st.set_page_config(page_title="ML Assignment Hub", layout="wide")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the project:", ["Student Segmentation", "Anime Recommender"])

if app_mode == "Student Segmentation":
    st.title("🎯 Student Segmentation (Clustering)")
    
    if not os.path.exists("models/clustering_artifacts.pkl"):
        st.error("Model artifacts not found. Please run 'python -m src.train_model' first.")
    else:
        with open("models/clustering_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        model = artifacts["model"]
        scaler = artifacts["scaler"]
        transformer = artifacts["transformer"]
        features = artifacts["features"]

        st.write("### Predict Student Cluster")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=50, value=18)
            grad_year = st.number_input("Graduation Year", min_value=2010, max_value=2030, value=2014)
        with col2:
            friends = st.number_input("Number of Friends", min_value=0, max_value=5000, value=100)

        st.write("#### Interests")
        user_interests = {}
        demographic_cols = {'age', 'numberoffriends', 'gradyear'}
        interest_cols = [f for f in features if f.lower() not in demographic_cols]
        
        cols = st.columns(4)
        for i, feat in enumerate(interest_cols[:12]): # Show first 12 interests for brevity
            user_interests[feat] = cols[i % 4].number_input(f"{feat.capitalize()}", min_value=0, max_value=10, value=0)

        if st.button("Classify Student"):
            # Initialize feature vector
            input_data = np.zeros((1, len(features)))
            feature_list = [f.lower() for f in features]
            
            # Map standard fields
            for col_name, val in [('age', age), ('numberoffriends', friends), ('gradyear', grad_year)]:
                if col_name in feature_list: 
                    input_data[0, feature_list.index(col_name)] = val
            
            # Map interest scores to their respective indices
            for feat, val in user_interests.items():
                if feat.lower() in feature_list:
                    input_data[0, feature_list.index(feat.lower())] = val
            
            # Create DataFrame to maintain feature names and prevent sklearn warnings
            input_df = pd.DataFrame(input_data, columns=features)
            input_transformed = transformer.transform(input_df)
            input_scaled = scaler.transform(input_transformed)
            
            prediction = model.predict(input_scaled)
            st.balloons()
            st.success(f"This student belongs to **Cluster {prediction[0]}**")

elif app_mode == "Anime Recommender":
    st.title("📺 Anime Recommendation System")
    
    def download_anime_data():
        try:
            import kaggle
            st.info("Anime data missing. Downloading from Kaggle...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files("CooperUnion/anime-recommendations-database", path=".", unzip=True)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download anime data: {e}")

    # Robust path check for anime data
    possible_paths = ["anime.csv", "data/raw/anime.csv", "data/anime.csv"]
    anime_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not anime_path:
        download_anime_data()
        anime_path = "anime.csv" if os.path.exists("anime.csv") else None

    if anime_path:
        @st.cache_resource
        def load_recommender(path):
            df = pd.read_csv(path)
            return AnimeRecommender(df)
        
        recommender = load_recommender(anime_path)
        
        # Sort list for better user experience
        anime_list = sorted(list(recommender.indices.index))
        anime_name = st.selectbox("Search for an Anime:", ["Select..."] + anime_list)
        
        if anime_name != "Select...":
            st.write(f"### Recommendations for {anime_name}:")
            recommendations = recommender.recommend(anime_name)
            
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
    else:
        st.error("anime.csv not found. Please ensure it is available or configured via Kaggle API.")