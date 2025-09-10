import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_neural_cf(num_users, num_items, embedding_dim=50):
    user_input = keras.Input(shape=(1,), name="user")
    item_input = keras.Input(shape=(1,), name="item")

    user_embedding = layers.Embedding(num_users, embedding_dim)(user_input)
    item_embedding = layers.Embedding(num_items, embedding_dim)(item_input)

    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    x = layers.Concatenate()([user_vec, item_vec])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1)(x)

    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_neural_cf(ratings, epochs=5, batch_size=256):
    ratings, user2idx, movie2idx = ratings  # unpack preprocessed ratings

    num_users = len(user2idx)
    num_items = len(movie2idx)

    model = build_neural_cf(num_users, num_items)

    X = [ratings["userId"].values, ratings["movieId"].values]
    y = ratings["rating"].values

    # ✅ Capture history
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Save trained model
    model.save("neural_cf_model.h5")

    return model, history
