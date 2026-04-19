import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AnimeRecommender:
    def __init__(self, anime_df):
        self.df = anime_df.copy()
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Prepares the data for the recommendation engine.

        Fills missing values for genre, type, and rating with appropriate values.
        Creates a content feature column by concatenating genre, type, and rating.
        Creates a TF-IDF matrix from the content feature column.
        Calculates the cosine similarity between the TF-IDF matrix and itself.
        Creates a lookup table for the search box used in the Streamlit UI.

        Returns:
            None
        """
        
        self.df['genre'] = self.df['genre'].fillna('')
        self.df['type'] = self.df['type'].fillna('')
        self.df['rating'] = self.df['rating'].fillna(0)
        
        self.df['content'] = (
            self.df['genre'] + " " + 
            self.df['type'] + " " + 
            self.df['rating'].astype(str)
        )
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['name']).drop_duplicates()

    def recommend(self, title, top_n=10):
        if title not in self.indices:
            return "Anime not found"
        
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        return self.df['name'].iloc[[i[0] for i in sim_scores]]