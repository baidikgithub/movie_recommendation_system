import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def combine_features(row):
    return f"{row['director_name']} {row['actor_1_name']} {row['actor_2_name']} {row['actor_3_name']} {row['genres']}"

def train_model(processed_file, model_path):
    df = pd.read_csv(processed_file)
    df['combined_features'] = df.apply(combine_features, axis=1)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(df['combined_features'])

    # Compute cosine similarity
    similarity = cosine_similarity(feature_matrix)

    # Save model & data
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump((df, similarity), f)

    print(f"Model trained & saved at {model_path}")

if __name__ == "__main__":
    train_model("data/processed/movies_processed.csv", "models/recommender.pkl")
