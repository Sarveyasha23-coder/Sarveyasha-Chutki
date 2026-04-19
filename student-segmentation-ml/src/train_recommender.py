import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.anime_recommender import load_anime_data, preprocess_anime_data, build_recommender, recommend_anime, plot_anime_rating_distribution, plot_top_genres, plot_top_popular_anime

def main():
    # File paths
    anime_path = "anime.csv"
    rating_path = "rating.csv"
    
    if not os.path.exists(anime_path) or not os.path.exists(rating_path):
        print("Please ensure anime.csv and rating.csv are in the root directory.")
        return

    # Load and Preprocess
    print("Loading Anime Data...")
    raw_anime, raw_ratings = load_anime_data(anime_path, rating_path)
    anime_df, ratings_df, merged_df = preprocess_anime_data(raw_anime, raw_ratings)

    # Part (c) iii: Skewness Check
    print("\n--- (c) iii: Anime Numeric Skewness ---")
    numeric_cols = anime_df.select_dtypes(include=['float64', 'int64']).columns
    print(anime_df[numeric_cols].skew())

    # Part (c) ii: EDA
    os.makedirs("plots/anime_eda", exist_ok=True)
    sns.set_style("whitegrid")
    
    # Using specialized plotting functions from anime_recommender.py
    plot_anime_rating_distribution(anime_df)
    plt.savefig("plots/anime_eda/rating_dist.png")
    plt.close()
    plot_top_genres(anime_df)
    plt.savefig("plots/anime_eda/top_genres.png")
    plt.close()
    plot_top_popular_anime(anime_df)
    plt.savefig("plots/anime_eda/top_popularity.png")
    plt.close()
    plt.close('all')

    # Part (c) iv: Build Recommender
    print("Building Content-Based Recommender System...")
    artifacts = build_recommender(anime_df)
    
    # Test Recommendation
    test_anime = "Naruto"
    print(f"\nRecommendations for '{test_anime}':")
    try:
        recs = recommend_anime(test_anime, artifacts.anime_content, artifacts.cosine_sim_matrix)
        print(recs[['name', 'similarity_score', 'genre']])
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
