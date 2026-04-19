import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from src.data_preprocessing import load_data, preprocess_data, get_skewness

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Map custom environment variables to standard Kaggle keys for deployment environments
if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_TOKEN" in os.environ:
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]
if "KAGGLE_USERNAME" not in os.environ and "KAGGLE_USER" in os.environ:
    os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USER"]

if os.environ.get("KAGGLE_KEY"):
    print(f"Kaggle API Token detected for user: {os.environ.get('KAGGLE_USERNAME', 'Unknown')}")

def download_kaggle_dataset(dataset_slug):
    try:
        import kaggle
        print(f"Data file missing. Attempting to download '{dataset_slug}' from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=".", unzip=True)
        print("Dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Please ensure kaggle.json is configured correctly in your .kaggle folder.")

def main():
    # Ensure the path matches your local environment
    data_file = "03_Clustering_Marketing.csv"
    dataset_slug = "zabihullah18/students-social-network-profile-clustering"
    
    if os.path.exists(data_file):
        data_path = data_file
    elif os.path.exists(os.path.join("data", "raw", data_file)):
        data_path = os.path.join("data", "raw", data_file)
    else:
        download_kaggle_dataset(dataset_slug)
        if os.path.exists(data_file):
            data_path = data_file
        elif os.path.exists("03_Clustering_Marketing.csv"):
            # Catch cases where Kaggle unzips directly to root
            data_path = "03_Clustering_Marketing.csv"
        elif os.path.exists(os.path.join("data", "raw", data_file)):
            data_path = os.path.join("data", "raw", data_file)
        else:
            raise FileNotFoundError(
                f"Could not find '{data_file}' after Kaggle download attempt. "
                "Place the file in the project root or in data/raw/."
            )
    
    df = load_data(data_path)
    
    # Part (b) iii: Check Skewness (On raw data)
    print("\n--- (b) iii: Feature Skewness (Before Transformation) ---")
    skew_report = get_skewness(df)
    print(skew_report)

    # Part (b) ii: EDA & Visualizations
    os.makedirs("plots/eda", exist_ok=True)
    sns.set_style("whitegrid")
    
    # 0. Missing Values Plot (Before preprocessing)
    plt.figure(figsize=(10, 5))
    df.isnull().sum().plot(kind='bar')
    plt.title("Missing Values per Column")
    plt.xticks(rotation=45)
    plt.savefig("plots/eda/missing_values.png")
    plt.close()

    X_scaled, df, pt, scaler, features_list = preprocess_data(df)

    # 1. Distributions of Key Variables
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")
    plt.savefig("plots/eda/age_distribution.png")
    plt.close()

    # 2. Outlier detection before capping
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['numberoffriends'])
    plt.title("Outliers in Number of Friends")
    plt.savefig("plots/eda/friends_outliers.png")
    plt.close()
    
    # Histogram of Friends
    plt.figure(figsize=(10, 6))
    sns.histplot(df['numberoffriends'], kde=True, bins=30)
    plt.title("Distribution of Number of Friends")
    plt.savefig("plots/eda/friends_distribution.png")
    plt.close()

    # 3. Interest analysis (Top 10)
    interest_cols = [c for c in df.columns if c not in ['age', 'gender', 'gradyear', 'numberoffriends', 'userid']]
    if interest_cols:
        plt.figure(figsize=(12, 5))
        df[interest_cols].sum().sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title("Top 10 Student Interests")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/eda/top_interests.png")
        plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("plots/eda/correlation_heatmap.png")
    plt.close()

    # Gender Distribution
    if 'gender' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='gender', data=df)
        plt.title("Gender Distribution")
        plt.savefig("plots/eda/gender_distribution.png")
        plt.close()

    print(f"Model trained on {len(features_list)} features.")

    # K-Means
    km_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = km_model.fit_predict(X_scaled)
    km_score = silhouette_score(X_scaled, km_labels)

    # Hierarchical
    hi_model = AgglomerativeClustering(n_clusters=3)
    hi_labels = hi_model.fit_predict(X_scaled)
    hi_score = silhouette_score(X_scaled, hi_labels)

    # DBSCAN
    db_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
    mask = db_labels != -1
    db_score = silhouette_score(X_scaled[mask], db_labels[mask]) if mask.any() and len(np.unique(db_labels[mask])) > 1 else 0

    # Part (b) vi: Comparison Table
    results = pd.DataFrame({
        'Model': ['K-Means', 'Hierarchical', 'DBSCAN'],
        'Silhouette Score': [float(km_score), float(hi_score), float(db_score)]
    })
    print("\n--- (b) vi: Clustering Method Comparison ---")
    print(results)

    # Performance Visualization
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(results['Model'], results['Silhouette Score'].fillna(0), color=['blue', 'green', 'orange'])
    plt.title('Clustering Model Performance')
    plt.ylabel('Silhouette Score')
    plt.savefig('models/clustering_comparison.png')
    plt.close()

    # Part (b) v: Demographic Profiling & Trend Analysis
    df['cluster'] = km_labels
    print("\n--- (b) v: Demographic Profiling (Cluster Means) ---")
    profile = df.groupby('cluster').mean(numeric_only=True)
    print(profile)

    # Part (b) v: PCA Visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=km_labels, cmap='viridis', alpha=0.6)
    plt.title("Clusters Visualization (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('models/pca_clusters.png')
    plt.close()

    # Part (b) v: Trend Analysis
    if 'gradyear' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='gradyear', y='numberoffriends', hue='cluster', data=df)
        plt.title("Trend of Social Activity Over Time")
        plt.savefig('models/trend_analysis.png')
        plt.close()

    # Save model and preprocessing objects
    artifacts = {
        "model": km_model,
        "scaler": scaler,
        "transformer": pt,
        "features": features_list
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/clustering_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    plt.close('all')

if __name__ == "__main__":
    main()
