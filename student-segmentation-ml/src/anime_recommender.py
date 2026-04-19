import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AnimeRecommender:
    def __init__(self, df):
        df = df[['name', 'genre']].dropna()
        df['genre'] = df['genre'].fillna('')
        
        self.df = df
        self.indices = pd.Series(df.index, index=df['name']).drop_duplicates()

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['genre'])

        self.similarity = cosine_similarity(tfidf_matrix)

    def recommend(self, title, n=5):
        idx = self.indices.get(title)
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return [self.df['name'].iloc[i[0]] for i in scores]
