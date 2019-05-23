import os
import pickle
import random
from random import shuffle

import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing, feature_extraction, model_selection

ds_info = {
    "input_files": {
        "ratings": "u.data",
        "users": "u.user",
        "movies": "u.item"
    },

    "ratings_params": {
        "dataset_name": "ml_100k",
        "columns": ["uId", "mId", "score"],
        "usecols": [0, 1, 2],
        "sep": "\t"
    },

    "user_feature_params": {
        "dataset_name": "ml_100k",
        "feature_type": "user",
        "columns": ["uId", "age", "gender", "job"],
        "formats": ["null", "B", "C", "B"],
        "usecols": [0, 1, 2, 3],
        "sep": "|"
    },

    "movie_feature_params": {
        "dataset_name": "ml_100k",
        "feature_type": "movie",
        "columns": ["mId", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",
                    "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19"],
        "formats": ["null", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
                    "C", "C", "C", "C", "C", "C", "C", "C"],
        "usecols": [0, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "sep": "|"
    }
}

from .ds import Dataset


class MovielensDataset(Dataset):
    def __init__(self, ml_dir, *args, **kwargs):
        super(MovielensDataset, self).__init__(args, kwargs)
        self.ml_dir = ml_dir
        self.config = ds_info
        self.shape = None
        self.ratings = None
        self.graph = None
        self.original_graph = None
        self.id_map = None
        self.idv_id_map = None

    def process_ratings(self):
        self._get_ratings_dataframe()
        # Enumerate movies & users
        mids = np.unique(self.ratings["mId"])
        uids = np.unique(self.ratings["uId"])

        # Filter data and transform
        # this is important, since in the raw data, user_id and movie_id have similar values,
        # thus uid=1 and mid=1 will confuse the nx, and it consider it as 1 node. Thus we fix the uid by adding len(movies)
        # -> uid=1 will become uid=len(movies) + 1
        self._remap_ids(self.ratings.values, uids, mids)

        # Node ID map back to movie and user IDs
        movie_id_map = {i: "m_{}".format(mId) for i, mId in enumerate(mids)}
        user_id_map = {i + len(mids): "u_{}".format(uId) for i, uId in enumerate(uids)}
        self.id_map = {**movie_id_map, **user_id_map}
        self.inv_id_map = dict(zip(self.id_map.values(), self.id_map.keys()))

        # Read graph
        print("Reading graph...")
        self.graph = nx.from_pandas_edgelist(self.ratings,
                                             source='uId',
                                             target='mId',
                                             edge_attr=True,
                                             create_using=nx.Graph())

        # Add node types:
        node_types = {self.inv_id_map["m_" + str(v)]: "movie" for v in mids}
        node_types.update({self.inv_id_map["u_" + str(v)]: "user" for v in uids})
        nx.set_node_attributes(self.graph, name="label", values=node_types)

        self.original_graph = self.graph.copy()
        print(
            "Graph statistics: {} users, {} movies, {} ratings".format(
                sum([v[1]["label"] == "user" for v in self.graph.nodes(data=True)]),
                sum([v[1]["label"] == "movie" for v in self.graph.nodes(data=True)]),
                self.graph.number_of_edges(),
            )
        )

        # Read features
        print("Reading features...")
        user_features_df = self._ingest_features(node_type="users")
        movie_features_df = self._ingest_features(node_type="movies")
        # Prepare the user features for ML (movie features are already numeric and hence ML-ready):
        feature_names = ["age", "gender", "job"]
        feature_encoding = feature_extraction.DictVectorizer(sparse=False, dtype=int)
        feature_encoding.fit(user_features_df[feature_names].to_dict("records"))

        vectorize_user_features = feature_encoding.transform(
            user_features_df[feature_names].to_dict("records")
        )
        # Assume that the age can be used as a continuous variable and rescale it
        vectorize_user_features[:, 0] = preprocessing.scale(vectorize_user_features[:, 0])
        self.graph
        feature_size = 32
        u_features = np.hstack((vectorize_user_features, np.zeros(
            (vectorize_user_features.shape[0], feature_size - vectorize_user_features.shape[1]))))
        m_features = np.hstack(
            (movie_features_df, np.zeros((movie_features_df.shape[0], feature_size - movie_features_df.shape[1]))))
        # Put features back in DataFrame
        vectorized_user_features_df = pd.DataFrame(
            u_features, index=user_features_df.index, dtype="float64"
        )
        vectorized_movie_features_df = pd.DataFrame(
            m_features, index=movie_features_df.index, dtype="float64"
        )

        # Add the user and movie features to the graph:
        # add 1 to feature size beacuse there is log degree feature in the function _add_features_to_nodes
        self.shape = [self.ratings.shape[0], feature_size + 1]
        self.graph = self._add_features_to_nodes(self.graph, self.inv_id_map,
                                                 vectorized_user_features_df,
                                                 vectorized_movie_features_df, add_log_degree=True)
        self.original_graph = self._add_features_to_nodes(self.original_graph, self.inv_id_map,
                                                          user_features_df,
                                                          movie_features_df, add_log_degree=False)

    # return X features and y lables for fitting
    def get_X_y(self):
        X, y = np.zeros(shape=self.shape), np.zeros(shape=(self.shape[0], 1))
        count = 0
        for edge in self.graph.edges:
            X[count] = self.get_combine_attr_from_nodes(edge[0], edge[1])
            y[count] = self.graph[edge[0]][edge[1]]['score']
            count += 1
        return X, y

    def get_Xedges_y(self):
        y = []
        X_edges = []
        for edge in self.graph.edges:
            X_edges.append(edge)
            y.append(self.graph[edge[0]][edge[1]]['score'])

        return X_edges, y

    def get_combine_attr_from_nodes(self, node1, node2):
        # swap to make concat user first then movie
        n1 = None,
        n2 = None
        type1 = self.graph.nodes[node1]['label']
        type2 = self.graph.nodes[node2]['label']
        if type1 == type2:
            raise Exception("Error, same type when combine node")

        if type1 == 'user' and type2 == 'movie':
            n1 = node1
            n2 = node2
        if type1 == 'movie' and type2 == 'user':
            n1 = node2
            n2 = node1

        combine_attr = np.concatenate((self.graph.node[n1]['feature'], self.graph.node[n2]['feature']))

        return combine_attr

    def sample_nb_nodes_with_similar_score(self, target_node, score, size=3):
        nb_nodess = self.graph[target_node]
        results = []

        # quick hack to find nodes with relevant score
        # todo: improve this
        chosen_nodes1 = []
        chosen_nodes2 = []
        chosen_nodes3 = []
        chosen_nodes4 = []
        chosen_nodes5 = []
        for nb_node in nb_nodess:
            new_score = abs(self.graph[target_node][nb_node]['score'] - score)
            if new_score < 1:
                chosen_nodes1.append(nb_node)
            if new_score >= 1 and new_score < 2:
                chosen_nodes2.append(nb_node)
            if new_score >= 2 and new_score < 3:
                chosen_nodes3.append(nb_node)
            if new_score >= 3 and new_score < 4:
                chosen_nodes4.append(nb_node)
            if new_score >= 4 and new_score <= 5:
                chosen_nodes5.append(nb_node)

        shuffle(chosen_nodes1)
        shuffle(chosen_nodes2)
        shuffle(chosen_nodes3)
        shuffle(chosen_nodes4)
        shuffle(chosen_nodes5)
        results += chosen_nodes1
        results += chosen_nodes2
        results += chosen_nodes3
        results += chosen_nodes4
        results += chosen_nodes5

        return results[:size]

    def _get_ratings_dataframe(self):
        if self.ratings is not None:
            return self.ratings

        rating_file = os.path.join(self.ml_dir, self.config["input_files"]["ratings"])
        columns = self.config["ratings_params"]["columns"]
        usecols = self.config["ratings_params"]["usecols"]
        sep = self.config["ratings_params"]["sep"]
        header = self.config["ratings_params"].get("header")

        # Load the edgelist:
        self.ratings = pd.read_csv(
            rating_file,
            names=columns,
            sep=sep,
            header=header,
            usecols=usecols,
            engine="python",
            dtype="int",
        )

    def _remap_ids(self, data, uid_map, mid_map, uid_inx=0, mid_inx=1):
        """
        Remap user and movie IDs
        """
        Nm = mid_map.shape[0]
        Nu = uid_map.shape[0]
        for ii in range(data.shape[0]):
            mid = data[ii, mid_inx]
            uid = data[ii, uid_inx]

            new_mid = np.searchsorted(mid_map, mid)
            new_uid = np.searchsorted(uid_map, uid)

            if new_mid < 0:
                print(mid, new_mid)

            # Only map to index if found, else map to zero
            if new_uid < Nu and (uid_map[new_uid] == uid):
                data[ii, uid_inx] = new_uid + Nm
            else:
                data[ii, uid_inx] = -1
            data[ii, mid_inx] = new_mid

    def _ingest_features(self, node_type):
        """Ingest fatures for nodes of node_type"""
        filename = os.path.join(self.ml_dir, self.config["input_files"][node_type])

        if node_type == "users":
            parameters = self.config["user_feature_params"]
        elif node_type == "movies":
            parameters = self.config["movie_feature_params"]
        else:
            raise ValueError("Unknown node type {}".format(node_type))

        columns = parameters.get("columns")
        formats = parameters.get("formats")
        usecols = parameters.get("usecols")
        sep = parameters.get("sep", ",")
        feature_type = parameters.get("feature_type")
        dtype = parameters.get("dtype", "float32")
        header = parameters.get("header")

        data = pd.read_csv(
            filename,
            index_col=0,
            names=columns,
            sep=sep,
            header=header,
            usecols=usecols,
        )

        return data

    def _add_features_to_nodes(self, g, inv_id_map, user_features, movie_features, add_log_degree=True):
        """Add user and movie features to graph nodes"""

        movie_features_dict = {
            k: np.array(movie_features.loc[k]) for k in movie_features.index
        }
        user_features_dict = {
            k: np.array(user_features.loc[k]) for k in user_features.index
        }

        node_features = {}
        for v in movie_features.index:
            node_id = inv_id_map["m_" + str(v)]
            log_node_degree = np.log(g.degree(node_id))
            mv_node_feature = movie_features_dict[v]
            if add_log_degree:
                mv_node_feature = np.hstack((mv_node_feature, np.array(log_node_degree)))
            node_features.update({node_id: mv_node_feature})

        for v in user_features.index:
            node_id = inv_id_map["u_" + str(v)]
            log_node_degree = np.log(g.degree(node_id))
            us_node_feature = user_features_dict[v]
            if add_log_degree:
                us_node_feature = np.hstack((us_node_feature, np.array(log_node_degree)))
            node_features.update({node_id: us_node_feature})

        nx.set_node_attributes(g, name="feature", values=node_features)

        return g
