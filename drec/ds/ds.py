import numpy as np
import networkx as nx


class Dataset():
    def __init__(self, *args, **kwargs):
        pass

    def get_nodes_attr(self, nodes):
        nodes_attr_dict = {}
        for node in nodes:
            node_attr = self.graph.nodes[node]['feature']
            nodes_attr_dict[node] = node_attr
        return nodes_attr_dict

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
        rating = 'score'
        for edge in self.graph.edges:
            X_edges.append(edge)
            y.append(self.graph[edge[0]][edge[1]][rating])

        return X_edges, y
