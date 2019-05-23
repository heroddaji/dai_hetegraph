# testing
from collections import defaultdict

from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import SVD, SVDpp, SlopeOne, NMF, KNNBasic
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split
from surprise import Prediction

from collections import defaultdict
import networkx as nx
import numpy as np
import math


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def intra_list_similarity(predictions, ds_graph):
    user_dict = defaultdict(dict)
    for pred in predictions:
        user_dict[pred.uid][pred.iid] = [pred.r_ui, pred.est]

    user_score = {}
    threshold = 15
    count_user = 0
    for user, items_dict in user_dict.items():
        # each user, cal the user ILS score
        item_scores = {}
        for item1_str, scores in items_dict.items():
            # each item, cal sim score
            for item2_str, scores in items_dict.items():
                item1 = int(item1_str)
                item2 = int(item2_str)

                if item_scores.get(item1 + item2, None) is not None:
                    continue

                sim = 0
                if item1 == item2:
                    sim = 1
                else:
                    try:
                        sim = cal_sim(ds_graph, item1, item2)
                    except Exception as e:
                        continue
                item_scores[item1 + item2] = sim
        ils_score = sum([v for v in item_scores.values()]) / len(item_scores)
        # print(f'\t user:{user} has ILS score:{ils_score}')
        user_score[user] = ils_score
        # count_user += 1
        # if count_user >= threshold:
        #     break

    average_ils = sum([v for v in user_score.values()]) / len(user_score)
    return average_ils


def cal_sim(g, item1, item2, sim_type='cosine'):
    sim = 0
    common_ds_users = sorted(nx.common_neighbors(g, item1, item2))
    ru1_list = [g[u][item1]['score'] for u in common_ds_users]
    ru2_list = [g[u][item2]['score'] for u in common_ds_users]
    if sim_type == 'cosine':
        sim = cal_cosine_sim(ru1_list, ru2_list)
    if np.isnan(sim):
        sim = 0
    return sim


def cal_cosine_sim(ru1_list, ru2_list):
    ru1_array = np.array(ru1_list)
    ru2_array = np.array(ru2_list)
    numerator = np.sum(ru1_array * ru2_array)
    denominator = np.sqrt(np.sum(np.square(ru1_array))) * np.sqrt(np.sum(np.square(ru2_array)))
    return numerator / denominator


def evaluate_pred(ds_graph, predictions):
    rmse = accuracy.rmse(predictions)
    precisions, recalls = precision_recall_at_k(predictions, k=20, threshold=3.5)
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    ils_sim = intra_list_similarity(predictions, ds_graph)
    return (rmse, prec, rec, ils_sim)


def run(ds, ds_name='movielens'):
    data = None
    if ds_name == 'movielens':
        # data = Dataset.load_builtin(ds_name)
        df = ds.ratings
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['uId', 'mId', 'score']], reader)
    if ds_name == 'bookcrossing':
        df = ds.ratings.sample(frac=0.2)
        reader = Reader(rating_scale=(0, 10))
        data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'score']], reader)

    ds_graph = ds.graph
    trainset, testset = train_test_split(data, test_size=.33)
    algos = [SVD(), SlopeOne(), NMF(), KNNBasic(), SVDpp()]
    for algo in algos:
        print(f'processing {ds_name} for algo {algo.__class__.__name__}')
        algo.fit(trainset)
        predictions = algo.test(testset)

        rmse, prec, rec, ils_sim = evaluate_pred(ds_graph, predictions)
        with open(f'eval_{ds_name}.csv', 'a') as f:
            f.write(f'{ds_name}_{algo.__class__.__name__},rmse,{rmse},precision,{prec},recall,{rec},ils,{ils_sim}\n')
