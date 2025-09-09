import surprise
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split

def collaborative_filtering(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Matrix factorization
    algo = SVD()
    algo.fit(trainset)

    predictions = algo.test(testset)

    # Cross-validation RMSE
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3)
    return algo, results
