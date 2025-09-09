import pandas as pd

def load_movielens(path="data/raw/ml-latest-small"):
    ratings = pd.read_csv(f"{path}/ratings.csv")
    movies = pd.read_csv(f"{path}/movies.csv")
    return ratings, movies

def preprocess_ids(ratings):
    """Map original userId and movieId to consecutive indices for embeddings."""
    user_ids = ratings["userId"].unique()
    user2idx = {old: new for new, old in enumerate(user_ids)}
    ratings["userId"] = ratings["userId"].map(user2idx)

    movie_ids = ratings["movieId"].unique()
    movie2idx = {old: new for new, old in enumerate(movie_ids)}
    ratings["movieId"] = ratings["movieId"].map(movie2idx)

    return ratings, user2idx, movie2idx
