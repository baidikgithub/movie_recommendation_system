import pandas as pd
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Fill missing values
    for col in ['director_name','actor_1_name','actor_2_name','actor_3_name','genres','movie_title']:
        df[col] = df[col].fillna('unknown')

    # Lowercase titles for consistency
    df['movie_title'] = df['movie_title'].str.lower()

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved at {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data("data/raw/data.csv", "data/processed/movies_processed.csv")
