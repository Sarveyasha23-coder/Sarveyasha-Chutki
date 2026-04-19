# Student Segmentation and Anime Recommendation Project

This repository contains an assignment-ready end-to-end machine learning project with:

1. A theory note explaining eigenvalues, eigenvectors, and their role in PCA.
2. A clustering project on the Students' Social Network Profile dataset.
3. A content-based recommendation project on the Anime Recommendations dataset.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks
│   ├── 01_eigenvalues_eigenvectors_pca.md
│   ├── 02_student_social_network_clustering.ipynb
│   └── 03_anime_recommendation_system.ipynb
├── src
│   ├── __init__.py
│   ├── student_clustering.py
│   └── anime_recommender.py
└── data
    ├── raw
    └── processed
```

## Datasets

The notebooks are designed for these Kaggle datasets:

1. Students' Social Network Profile Clustering
   `https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering/data`
2. Anime Recommendations Database
   `https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database`

Place the downloaded files in `data/raw/`:

- `03_Clustering_Marketing.csv`
- `anime.csv`
- `rating.csv`

## Install

```bash
pip install -r requirements.txt
```

## Run in Jupyter

```bash
jupyter notebook
```

Open:

1. `notebooks/02_student_social_network_clustering.ipynb`
2. `notebooks/03_anime_recommendation_system.ipynb`

## Run in Colab

You can upload the notebooks to Colab and then:

1. Upload the dataset files into the Colab session, or mount Google Drive.
2. Adjust the dataset paths if needed.
3. Install the required packages with:

```python
!pip install -q pandas numpy matplotlib seaborn scikit-learn scipy kaggle
```

If you want to use Kaggle directly in Colab, upload `kaggle.json` and configure it in the notebook runtime before downloading the datasets.

## Kaggle Download Setup

If you are running locally:

1. Create a Kaggle API token from your Kaggle account.
2. Save `kaggle.json`.
3. Put it in your Kaggle config folder:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then you can download:

```bash
kaggle datasets download -d zabihullah18/students-social-network-profile-clustering
kaggle datasets download -d CooperUnion/anime-recommendations-database
```

## Notebooks

1. `notebooks/01_eigenvalues_eigenvectors_pca.md`
   Theory explanation for eigenvalues, eigenvectors, examples, and PCA.
2. `notebooks/02_student_social_network_clustering.ipynb`
   End-to-end clustering workflow with EDA, preprocessing, PCA visualization, clustering, and comparison.
3. `notebooks/03_anime_recommendation_system.ipynb`
   End-to-end anime recommendation workflow with EDA and content-based recommendations.

## Notes

- The clustering notebook compares K-Means, Hierarchical Clustering, and DBSCAN using silhouette score where valid.
- The student project includes demographic profiling and interest-based cluster interpretation.
- The anime project uses TF-IDF over genre, type, and rating-based content features.
- The code is written so it can be reused from Jupyter, Colab, or regular Python scripts.
- If you already have the CSV files locally, you do not need to use the Kaggle API cells.
