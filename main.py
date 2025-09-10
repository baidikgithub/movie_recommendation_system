from utils.preprocess import load_movielens, preprocess_ids
from models.content_based import ContentBasedRecommender
from models.neural_cf import train_neural_cf, build_neural_cf
import pandas as pd
import matplotlib.pyplot as plt
import json

def main():
    # ------------------------------
    # Load MovieLens data
    # ------------------------------
    ratings, movies = load_movielens("data/raw/ml-latest-small")

    # ------------------------------
    # Content-Based Recommendations
    # ------------------------------
    cb = ContentBasedRecommender(movies)
    print("Content-based Recommendations for 'Toy Story (1995)':")
    print(cb.recommend("Toy Story (1995)", top_k=5))

    # ------------------------------
    # Neural Collaborative Filtering
    # ------------------------------
    print("\nPreprocessing IDs for Neural CF...")
    ratings, user2idx, movie2idx = preprocess_ids(ratings)

    print("Training Neural Collaborative Filtering model...")
    model, history = train_neural_cf((ratings, user2idx, movie2idx), epochs=5)
    print("Neural CF training complete!")

    # ------------------------------
    # Save training history for notebook visualization
    # ------------------------------
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)


if __name__ == "__main__":
    main()
