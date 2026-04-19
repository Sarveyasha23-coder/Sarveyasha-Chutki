from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AnimeRecommenderArtifacts:
    anime: pd.DataFrame
    ratings: pd.DataFrame
    merged: pd.DataFrame
    anime_content: pd.DataFrame
    tfidf_matrix: object
    cosine_sim_matrix: object


def load_anime_data(anime_path: str, rating_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    anime = pd.read_csv(anime_path)
    ratings = pd.read_csv(rating_path)
    return anime, ratings


def preprocess_anime_data(anime: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anime_df = anime.copy()
    ratings_df = ratings.copy()

    anime_df.columns = anime_df.columns.str.lower().str.strip()
    ratings_df.columns = ratings_df.columns.str.lower().str.strip()

    if "episodes" in anime_df.columns:
        anime_df["episodes"] = pd.to_numeric(anime_df["episodes"], errors="coerce")
    if "rating" in anime_df.columns:
        anime_df["rating"] = pd.to_numeric(anime_df["rating"], errors="coerce")
    if "members" in anime_df.columns:
        anime_df["members"] = pd.to_numeric(anime_df["members"], errors="coerce")

    ratings_df = ratings_df[ratings_df["rating"] != -1].copy()
    merged = ratings_df.merge(anime_df, on="anime_id", how="inner", suffixes=("_user", "_anime"))
    return anime_df, ratings_df, merged


def build_content_features(anime_df: pd.DataFrame) -> pd.DataFrame:
    content_df = anime_df[["anime_id", "name", "genre", "type", "rating", "members"]].copy()
    content_df["genre"] = content_df["genre"].fillna("")
    content_df["type"] = content_df["type"].fillna("")
    content_df["rating"] = content_df["rating"].fillna(content_df["rating"].median() if "rating" in content_df else 0)
    content_df["members"] = content_df["members"].fillna(0)

    content_df["rating_bucket"] = pd.cut(
        content_df["rating"],
        bins=[-float("inf"), 5, 7, 8.5, float("inf")],
        labels=["low_rating", "mid_rating", "good_rating", "top_rating"],
    ).astype(str)

    content_df["popularity_bucket"] = pd.cut(
        content_df["members"],
        bins=[-float("inf"), 1e4, 1e5, 5e5, float("inf")],
        labels=["niche", "growing", "popular", "blockbuster"],
    ).astype(str)

    content_df["content"] = (
        content_df["genre"].astype(str)
        + " "
        + content_df["type"].astype(str)
        + " "
        + content_df["rating_bucket"].astype(str)
        + " "
        + content_df["popularity_bucket"].astype(str)
    )
    return content_df


def build_recommender(anime_df: pd.DataFrame) -> AnimeRecommenderArtifacts:
    anime_content = build_content_features(anime_df)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(anime_content["content"])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    empty_ratings = pd.DataFrame()
    empty_merged = pd.DataFrame()
    return AnimeRecommenderArtifacts(
        anime=anime_df,
        ratings=empty_ratings,
        merged=empty_merged,
        anime_content=anime_content,
        tfidf_matrix=tfidf_matrix,
        cosine_sim_matrix=cosine_sim_matrix,
    )


def recommend_anime(title: str, anime_content: pd.DataFrame, cosine_sim_matrix, top_n: int = 10) -> pd.DataFrame:
    index_lookup = pd.Series(anime_content.index, index=anime_content["name"])
    index_lookup = index_lookup[~index_lookup.index.duplicated(keep='first')]
    if title not in index_lookup:
        raise ValueError(f"'{title}' was not found in the dataset.")

    idx = index_lookup[title]
    similarity_scores = list(enumerate(cosine_sim_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)
    similarity_scores = similarity_scores[1 : top_n + 1]
    anime_indices = [i for i, _ in similarity_scores]

    result = anime_content.loc[anime_indices, ["name", "genre", "type", "rating", "members"]].copy()
    result.insert(0, "similarity_score", [score for _, score in similarity_scores])
    return result.reset_index(drop=True)


def plot_anime_missing_values(anime_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    anime_df.isna().sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Values in Anime Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


def plot_anime_rating_distribution(anime_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(anime_df["rating"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Anime Ratings")
    plt.tight_layout()


def plot_top_popular_anime(anime_df: pd.DataFrame, top_n: int = 10) -> None:
    top_anime = anime_df.sort_values(by="members", ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_anime, x="members", y="name")
    plt.title(f"Top {top_n} Most Popular Anime")
    plt.tight_layout()


def plot_anime_type_distribution(anime_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=anime_df, x="type", order=anime_df["type"].value_counts().index)
    plt.title("Distribution of Anime Types")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


def plot_episode_distribution(anime_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(anime_df["episodes"].dropna(), bins=50)
    plt.title("Distribution of Episodes")
    plt.tight_layout()


def plot_user_rating_distribution(ratings_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(ratings_df["rating"], bins=20)
    plt.title("User Rating Distribution")
    plt.tight_layout()


def plot_top_rated_anime(merged_df: pd.DataFrame, min_ratings: int = 100, top_n: int = 10) -> None:
    grouped = merged_df.groupby("name").agg(mean_user_rating=("rating_user", "mean"), rating_count=("rating_user", "count"))
    filtered = grouped[grouped["rating_count"] >= min_ratings].sort_values("mean_user_rating", ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=filtered["mean_user_rating"], y=filtered.index)
    plt.title(f"Top Rated Anime with at least {min_ratings} Ratings")
    plt.tight_layout()


def plot_most_rated_anime(merged_df: pd.DataFrame, top_n: int = 10) -> None:
    counts = merged_df["name"].value_counts().head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {top_n} Most Rated Anime")
    plt.tight_layout()


def plot_anime_correlation_heatmap(anime_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.heatmap(anime_df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()


def plot_top_genres(anime_df: pd.DataFrame, top_n: int = 10) -> None:
    genre_counts = anime_df["genre"].dropna().str.split(",").explode().str.strip().value_counts().head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title(f"Top {top_n} Anime Genres")
    plt.tight_layout()


def plot_user_activity(ratings_df: pd.DataFrame) -> None:
    user_activity = ratings_df["user_id"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.histplot(user_activity, bins=50)
    plt.title("User Activity Distribution")
    plt.tight_layout()


class AnimeRecommender:
    """Wrapper class for the Streamlit app to interact with the recommendation logic."""
    def __init__(self, df: pd.DataFrame):
        # Standardize columns and types to avoid errors during build_recommender
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        for col in ["episodes", "rating", "members"]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        self.artifacts = build_recommender(df_clean)
        # Create a lookup for the search box used in the Streamlit UI
        self.indices = pd.Series(
            self.artifacts.anime_content.index, 
            index=self.artifacts.anime_content["name"]
        ).drop_duplicates()

    def recommend(self, title: str, top_n: int = 10) -> List[str] | str:
        try:
            results = recommend_anime(title, self.artifacts.anime_content, self.artifacts.cosine_sim_matrix, top_n)
            return results["name"].tolist()
        except Exception as e:
            return str(e)
