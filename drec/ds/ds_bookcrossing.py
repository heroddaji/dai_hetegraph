import os
from random import shuffle

import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing, feature_extraction
from sklearn.feature_extraction.text import HashingVectorizer

from .ds import Dataset

ds_info = {
    "input_files": {
        "ratings": "BX-Book-Ratings.csv",
        "books": "BX-Books.csv",
        "users": "BX-Users.csv"
    },

    "ratings_params": {
        "dataset_name": "BX-CSV-Dump",
        "columns": ["User-ID", "ISBN", "Book-Rating"],
        "usecols": [0, 1, 2],
        "sep": ";"
    },

    "user_feature_params": {
        "dataset_name": "BX-CSV-Dump",
        "feature_type": "user",
        "columns": ["User-ID", "Location", "Age"],
        "formats": ["null", "B", "C", "B"],
        "usecols": [0, 1, 2],
        "sep": ";"
    },

    "book_feature_params": {
        "dataset_name": "BX-CSV-Dump",
        "feature_type": "book",
        "columns": ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S",
                    "Image-URL-M", "Image-URL-L"],
        "formats": ["null", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
                    "C", "C", "C", "C", "C", "C", "C", "C"],
        "usecols": [0, 1, 2, 3, 4, 5, 6, 7],
        "sep": ";"
    }
}


class BookCrossingDataset(Dataset):
    def __init__(self, ds_dir, *args, **kwargs):
        super(BookCrossingDataset, self).__init__(args, kwargs)
        self.ds_dir = ds_dir
        self.config = ds_info
        self.shape = None
        self.ratings = None
        self.graph = None
        self.newId_oldId_map = None
        self.oldId_newId_map = None

    def process_ratings(self):
        self._get_ratings_dataframe()
        feature_size = 64
        user_features = self._ingest_features(node_type="users")
        user_features['Age'].fillna(0, inplace=True)
        book_features = self._ingest_features(node_type="books")

        # remap the id of both items and users
        ratingBookIds = np.unique(self.ratings['ISBN'])
        ratingUserIds = np.unique(self.ratings['User-ID'])
        originBookIds = np.concatenate((np.unique(book_features.index), ratingBookIds))
        originUserIds = np.concatenate((np.unique(user_features.index), ratingUserIds))
        newId_originId_books = dict(zip([i for i in range(len(originBookIds))], originBookIds))
        newId_originId_users = dict(zip([i + len(originBookIds) for i in range(len(originBookIds))], originUserIds))

        self.newId_oldId_map = {**newId_originId_books, **newId_originId_users}
        self.oldId_newId_map = dict(zip(self.newId_oldId_map.values(), self.newId_oldId_map.keys()))

        # apply new id to the rating
        self.ratings['ISBN'] = self.ratings['ISBN'].apply(lambda x: self.oldId_newId_map[x])
        self.ratings['User-ID'] = self.ratings['User-ID'].apply(lambda x: self.oldId_newId_map[x])

        # change "rating" column to "score"
        self.ratings.rename(columns={'Book-Rating': 'score'}, inplace=True)

        # Read graph
        print("Reading graph...")
        self.graph = nx.from_pandas_edgelist(self.ratings,
                                             source='User-ID',
                                             target='ISBN',
                                             edge_attr=True,
                                             create_using=nx.Graph())
        # Add node types:
        node_types = {self.oldId_newId_map[v]: "book" for v in ratingBookIds}
        node_types.update({self.oldId_newId_map[v]: "user" for v in ratingUserIds})
        nx.set_node_attributes(self.graph, name="label", values=node_types)

        print(
            "Graph statistics of rating files: {} users, {} books, {} ratings".format(
                sum([v[1]["label"] == "user" for v in self.graph.nodes(data=True)]),
                sum([v[1]["label"] == "book" for v in self.graph.nodes(data=True)]),
                self.graph.number_of_edges(),
            )
        )

        print("Transforming features...")
        # vectorize user and book features
        user_feature_names = ["Location", "Age"]

        # make user location become vector
        strHashVectorizer = HashingVectorizer(n_features=2 ** 4)
        loc_transformed = strHashVectorizer.fit_transform(user_features[user_feature_names[0]]).toarray()
        # make user age become vector
        age_array = user_features[user_feature_names[1]].values
        # Assume that the age can be used as a continuous variable and rescale it
        age_transformed = preprocessing.scale(age_array)[:, None]

        # Put vectorized user features back to dataframe
        u_features = np.hstack((loc_transformed, age_transformed))
        u_features = np.hstack((u_features, np.zeros((u_features.shape[0], feature_size - u_features.shape[1]))))

        user_features = pd.DataFrame(
            u_features, index=user_features.index, dtype="float64"
        )

        book_feature_names = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
        book_features[book_feature_names[1]].fillna("unknown", inplace=True)
        book_features[book_feature_names[2]].fillna('0', inplace=True)
        book_features[book_feature_names[2]] = book_features[book_feature_names[2]].astype('str')
        book_features[book_feature_names[3]].fillna("unknown", inplace=True)
        # vectorize book titles
        strHashVectorizer = HashingVectorizer(n_features=2 ** 4)
        booktitle_transformed = strHashVectorizer.fit_transform(book_features[book_feature_names[0]]).toarray()
        # vectorize book authors
        strHashVectorizer = HashingVectorizer(n_features=2 ** 4, ngram_range=(1, 4))
        bookauthor_transformed = strHashVectorizer.fit_transform(
            book_features[book_feature_names[1]]).toarray()
        # rescale year value
        strHashVectorizer = HashingVectorizer(n_features=2 ** 3)
        year_transformed = strHashVectorizer.fit_transform(
            book_features[book_feature_names[2]]).toarray()
        # vectorize book publisher
        strHashVectorizer = HashingVectorizer(n_features=2 ** 4)
        bookpublisher_transformed = strHashVectorizer.fit_transform(
            book_features[book_feature_names[3]]).toarray()

        # todo: make same size feature for both book and user

        b_features = np.hstack(
            (booktitle_transformed, bookauthor_transformed, year_transformed, bookpublisher_transformed))
        b_features = np.hstack((b_features, np.zeros((b_features.shape[0], feature_size - b_features.shape[1]))))
        # Put book features back in DataFrame
        book_features = pd.DataFrame(
            b_features,
            index=book_features.index, dtype="float64"
        )

        # Add the user and movie features to the graph:
        print("Update graph with vectorized features...")
        self.shape = [self.ratings.shape[0], user_features.shape[
            1] + 1]  # add 1 to feature size beacuse there is log degree feature in the function _add_features_to_nodes
        self._add_features_to_nodes(user_features, book_features, add_log_degree=True)

    def get_combine_attr_from_nodes(self, node1, node2):
        # swap to make concat user first then movie
        combine = None
        try:
            n1 = None,
            n2 = None
            type1 = self.graph.nodes[node1]['label']
            type2 = self.graph.nodes[node2]['label']
            if type1 == type2:
                raise Exception("Error, same type when combine node")

            if type1 == 'user' and type2 == 'book':
                n1 = node1
                n2 = node2
            if type1 == 'book' and type2 == 'user':
                n1 = node2
                n2 = node1

            combine = np.concatenate((self.graph.node[n1]['feature'], self.graph.node[n2]['feature']))
        except Exception as e:
            print(e)

        return combine

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
        chosen_nodes6 = []
        chosen_nodes7 = []
        chosen_nodes8 = []
        chosen_nodes9 = []
        chosen_nodes10 = []
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
            if new_score >= 4 and new_score < 5:
                chosen_nodes5.append(nb_node)
            if new_score >= 5 and new_score < 6:
                chosen_nodes6.append(nb_node)
            if new_score >= 6 and new_score < 7:
                chosen_nodes7.append(nb_node)
            if new_score >= 7 and new_score < 8:
                chosen_nodes8.append(nb_node)
            if new_score >= 8 and new_score < 9:
                chosen_nodes9.append(nb_node)
            if new_score >= 9 and new_score <= 10:
                chosen_nodes10.append(nb_node)

        shuffle(chosen_nodes1)
        shuffle(chosen_nodes2)
        shuffle(chosen_nodes3)
        shuffle(chosen_nodes4)
        shuffle(chosen_nodes5)
        shuffle(chosen_nodes6)
        shuffle(chosen_nodes7)
        shuffle(chosen_nodes8)
        shuffle(chosen_nodes9)
        shuffle(chosen_nodes10)
        results += chosen_nodes1
        results += chosen_nodes2
        results += chosen_nodes3
        results += chosen_nodes4
        results += chosen_nodes5
        results += chosen_nodes6
        results += chosen_nodes7
        results += chosen_nodes8
        results += chosen_nodes9
        results += chosen_nodes10

        return results[:size]

    def _get_ratings_dataframe(self):
        if self.ratings is not None:
            return self.ratings

        rating_file = os.path.join(self.ds_dir, self.config["input_files"]["ratings"])
        columns = self.config["ratings_params"]["columns"]
        usecols = self.config["ratings_params"]["usecols"]
        sep = self.config["ratings_params"]["sep"]
        header = self.config["ratings_params"].get("header")

        # Load the edgelist:
        self.ratings = pd.read_csv(
            rating_file,
            names=columns,
            sep=sep,
            header=0,
            usecols=usecols,
            encoding="ISO-8859-1",
            # encoding="utf-8",
        )
        self.ratings["Book-Rating"] = pd.to_numeric(self.ratings["Book-Rating"])
        print(self.ratings.info())

    def _ingest_features(self, node_type):
        """Ingest fatures for nodes of node_type"""
        filename = os.path.join(self.ds_dir, self.config["input_files"][node_type])

        if node_type == "users":
            parameters = self.config["user_feature_params"]
        elif node_type == "books":
            parameters = self.config["book_feature_params"]
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
            header=0,
            usecols=usecols,
            encoding="ISO-8859-1",
        )

        return data

    def _add_features_to_nodes(self, user_features, item_features, add_log_degree=True):
        """Add user and item features to graph nodes"""
        none_feature_nodes = []
        # loop through all nodes in graph, if any node is missing feature, remove that node
        for node in self.graph.nodes(data=True):
            node_id = node[0]
            node_label = node[1]['label']
            old_nodeId = self.newId_oldId_map[node_id]
            feature = None
            try:
                if node_label == 'user':
                    feature = user_features.loc[old_nodeId]
                if node_label == 'book':
                    feature = item_features.loc[old_nodeId]
                log_node_degree = np.log(self.graph.degree(node_id))
                if add_log_degree:
                    feature = np.hstack((feature, np.array(log_node_degree)))
                node_feature = {node_id: {'feature': feature}}

                nx.set_node_attributes(self.graph, node_feature)
            except Exception as e:
                none_feature_nodes.append(node_id)

        print(f'before clean, edges count:{self.graph.size()}')

        # remove nodes without features
        self.graph.remove_nodes_from(none_feature_nodes)

        print(f"removed {len(none_feature_nodes)} nodes")
        print(f'AFTER clean, edgeS count:{self.graph.size()}')
