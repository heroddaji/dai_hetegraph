import numpy as np
import utils
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

def run_svd(dataset):


    # Load the movielens_hetesage-100k dataset (download it if needed),
    data = Dataset.load_builtin(dataset)

    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainset, testset = train_test_split(data, test_size=.33)

    # We'll use the famous SVD algorithm.
    algo = SVD()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    accuracy.rmse(predictions)
    y_test = [item[2] for item in testset]
    preds = [pred[3] for pred in predictions]
    preds_round = np.rint(preds)
    rmse_round = np.sqrt(np.mean(np.square(np.array(preds_round - np.array(y_test)))))
    print(f'rmse_round {rmse_round}')
    utils.hist_plot(y_test, preds, preds_round)
