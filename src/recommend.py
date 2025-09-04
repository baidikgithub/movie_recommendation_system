import pickle

def recommend(movie_title, model_path="models/recommender.pkl", top_n=5):
    with open(model_path, "rb") as f:
        df, similarity = pickle.load(f)

    movie_title = movie_title.lower()

    if movie_title not in df['movie_title'].values:
        print(f"❌ '{movie_title}' not found in dataset.")
        return []

    idx = df[df['movie_title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended_movies = [df.iloc[i[0]]['movie_title'] for i in sorted_scores]
    return recommended_movies

if __name__ == "__main__":
    movie = input("🎬 Enter a movie title: ")
    recs = recommend(movie)
    print("\n✅ Recommended movies similar to", movie, ":")
    for r in recs:
        print("👉", r)
