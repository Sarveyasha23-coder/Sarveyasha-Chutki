from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


@dataclass
class StudentClusteringArtifacts:
    cleaned_data: pd.DataFrame
    feature_matrix: np.ndarray
    feature_names: List[str]
    kmeans_labels: np.ndarray
    hierarchical_labels: np.ndarray
    dbscan_labels: np.ndarray
    model_scores: pd.DataFrame
    pca_components: np.ndarray
    cluster_profile: pd.DataFrame


def load_student_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    for col in ["age", "numberoffriends", "gradyear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def get_student_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    id_cols = [col for col in df.columns if col in {"userid"}]
    categorical_cols = [col for col in ["gender"] if col in df.columns]
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in id_cols]
    return numeric_cols, categorical_cols, id_cols


def cap_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    capped = df.copy()
    for col in columns:
        series = capped[col]
        if series.dropna().empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped[col] = series.clip(lower=lower, upper=upper)
    return capped


def skewness_report(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.Series:
    if numeric_cols is None:
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
    return df[numeric_cols].skew().sort_values(ascending=False)


def preprocess_student_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], ColumnTransformer]:
    working = df.copy()
    numeric_cols, categorical_cols, id_cols = get_student_column_groups(working)

    if numeric_cols:
        working = cap_outliers_iqr(working, numeric_cols)

    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("power", PowerTransformer(method="yeo-johnson")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    feature_matrix = transformer.fit_transform(working)

    feature_names: List[str] = []
    feature_names.extend(numeric_cols)
    if categorical_cols:
        encoder = transformer.named_transformers_["cat"].named_steps["encoder"]
        feature_names.extend(list(encoder.get_feature_names_out(categorical_cols)))

    cleaned = working.drop(columns=id_cols, errors="ignore")
    return cleaned, feature_matrix, feature_names, transformer


def elbow_method(features: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(features)
        rows.append({"k": k, "inertia": model.inertia_})
    return pd.DataFrame(rows)


def _safe_silhouette_score(features: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    return float(silhouette_score(features, labels))


def _safe_dbscan_silhouette(features: np.ndarray, labels: np.ndarray) -> Optional[float]:
    mask = labels != -1
    filtered = labels[mask]
    if mask.sum() == 0 or len(np.unique(filtered)) < 2:
        return None
    return float(silhouette_score(features[mask], filtered))


def compare_clustering_models(
    features: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    dbscan_eps: float = 2.2,
    dbscan_min_samples: int = 8,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(features)

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(features)

    results = pd.DataFrame(
        [
            {
                "model": "K-Means",
                "silhouette_score": _safe_silhouette_score(features, kmeans_labels),
                "clusters_found": int(len(np.unique(kmeans_labels))),
            },
            {
                "model": "Hierarchical",
                "silhouette_score": _safe_silhouette_score(features, hierarchical_labels),
                "clusters_found": int(len(np.unique(hierarchical_labels))),
            },
            {
                "model": "DBSCAN",
                "silhouette_score": _safe_dbscan_silhouette(features, dbscan_labels),
                "clusters_found": int(len(np.unique(dbscan_labels[dbscan_labels != -1]))),
            },
        ]
    )

    label_map = {
        "kmeans": kmeans_labels,
        "hierarchical": hierarchical_labels,
        "dbscan": dbscan_labels,
    }
    return label_map, results


def create_cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profiled = df.copy()
    profiled["cluster"] = labels
    numeric_cols = list(profiled.select_dtypes(include=np.number).columns)
    return profiled.groupby("cluster")[numeric_cols].mean().round(3)


def run_student_clustering_project(
    csv_path: str,
    n_clusters: int = 3,
    random_state: int = 42,
) -> StudentClusteringArtifacts:
    raw_df = load_student_data(csv_path)
    cleaned_df, features, feature_names, _ = preprocess_student_data(raw_df)
    label_map, model_scores = compare_clustering_models(
        features=features,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    pca = PCA(n_components=2, random_state=random_state)
    pca_components = pca.fit_transform(features)
    cluster_profile = create_cluster_profile(cleaned_df, label_map["kmeans"])

    return StudentClusteringArtifacts(
        cleaned_data=cleaned_df.assign(cluster=label_map["kmeans"]),
        feature_matrix=features,
        feature_names=feature_names,
        kmeans_labels=label_map["kmeans"],
        hierarchical_labels=label_map["hierarchical"],
        dbscan_labels=label_map["dbscan"],
        model_scores=model_scores,
        pca_components=pca_components,
        cluster_profile=cluster_profile,
    )


def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 4)) -> None:
    plt.figure(figsize=figsize)
    df.isna().sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Values per Column")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, columns: List[str], bins: int = 30) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=bins)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


def plot_boxplots(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_interest_totals(df: pd.DataFrame, top_n: int = 12) -> None:
    interest_candidates = [
        col
        for col in df.columns
        if col not in {"userid", "gender", "age", "gradyear", "numberoffriends", "cluster"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not interest_candidates:
        return
    totals = df[interest_candidates].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 4))
    totals.plot(kind="bar")
    plt.title(f"Top {top_n} Interest Totals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_elbow_curve(elbow_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS / Inertia")
    plt.tight_layout()
    plt.show()


def plot_pca_clusters(components: np.ndarray, labels: np.ndarray, title: str = "Cluster Visualization using PCA") -> None:
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.show()


def plot_gender_by_cluster(df: pd.DataFrame) -> None:
    if "gender" not in df.columns or "cluster" not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="cluster", hue="gender")
    plt.title("Gender Distribution by Cluster")
    plt.tight_layout()
    plt.show()


def plot_interest_by_cluster(df: pd.DataFrame, interest_cols: Optional[List[str]] = None, top_n: int = 8) -> None:
    if "cluster" not in df.columns:
        return
    if interest_cols is None:
        candidates = [
            col
            for col in df.columns
            if col not in {"userid", "gender", "age", "gradyear", "numberoffriends", "cluster"}
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        interest_cols = list(df[candidates].mean().sort_values(ascending=False).head(top_n).index)
    if not interest_cols:
        return
    df.groupby("cluster")[interest_cols].mean().plot(kind="bar", figsize=(10, 5))
    plt.title("Interest Patterns by Cluster")
    plt.ylabel("Average Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_social_activity_trend(df: pd.DataFrame) -> None:
    required = {"gradyear", "numberoffriends", "cluster"}
    if not required.issubset(df.columns):
        return
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x="gradyear", y="numberoffriends", hue="cluster", estimator="mean", errorbar=None)
    plt.title("Trend of Social Activity Over Time")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: pd.DataFrame) -> None:
    plotting = results.copy()
    plotting["silhouette_score"] = plotting["silhouette_score"].fillna(0)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plotting, x="model", y="silhouette_score")
    plt.title("Silhouette Score Comparison")
    plt.ylabel("Score")
    plt.xlabel("Clustering Method")
    plt.tight_layout()
    plt.show()
