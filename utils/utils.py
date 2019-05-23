import pickle
import matplotlib.pyplot as plt


def save_obj(obj, path):
    with open(path, 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        obj = None
    return obj


def get_unique_nodes_from_edges(edges):
    unique_nodes_s = set()
    [unique_nodes_s.update(node_tuple) for node_tuple in edges]
    unique_nodes = list(unique_nodes_s)
    return unique_nodes


def get_unique_nodes_from_nodes(node_pairs):
    return list(set(node_pairs))


def hist_plot(tests, preds, preds_round):
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.hist(tests, bins=50, range=(-1, 5))
    plt.subplot(132)
    plt.hist(preds, bins=50, range=(-1, 5))
    plt.subplot(133)
    plt.hist(preds_round, bins=50, range=(-1, 5))
    plt.show()
