import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies):
        self.df = movies.copy()
        self.df['movie_title'] = self.df['title'].str.lower()
        self.df['combined_features'] = self.df['genres']  # You can add director/actors if available

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        self.similarity = cosine_similarity(self.feature_matrix)

    def recommend(self, movie_title, top_k=5):
        movie_title = movie_title.lower()
        if movie_title not in self.df['movie_title'].values:
            return []

        idx = self.df[self.df['movie_title'] == movie_title].index[0]
        scores = list(enumerate(self.similarity[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]

        recommended_movies = [self.df.iloc[i[0]]['title'] for i in sorted_scores]
        return recommended_movies
