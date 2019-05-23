import torch
import torch.nn as  nn
import torch.nn.functional as F


class HeteEdgeAggregator(nn.Module):
    def __init__(self, dataset, encoder=None, **kwargs):
        super(HeteEdgeAggregator, self).__init__()
        self.dataset = dataset
        self.encoder = encoder
        self.layer = kwargs.get('layer', 1)

    def forward(self, edges):
        pass  # implement by subclass


class HeteEdgeMeanAggregator(HeteEdgeAggregator):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    kwarg values:
    - layer: a layer number to keep track (since there is many recursive calling of encoder and aggregator)
    """

    def __init__(self, *args, **kwargs):
        super(HeteEdgeMeanAggregator, self).__init__(*args, **kwargs)
        self._get_config()
        self.nb_walk_size = self.config['nb_walk_size']
        self.edges_and_nb_dict = {}  # store the edge and its nb after sampling

    def _get_config(self):
        self.config = {
            'type': 'mean',
            'nb_walk_size': 5
        }

    def forward(self, edges):
        g = self.dataset.graph
        edges_attr = []
        edges_nb_attr = []

        if self.encoder is None:
            # first aggregation layer, take the input feature directly from graph
            for edge in edges:
                # compute target_edge attr
                edge_score = g.edges[edge]['score']
                node1 = edge[0]
                node2 = edge[1]
                self_attr = torch.Tensor(self.dataset.get_combine_attr_from_nodes(node1, node2))
                edges_attr.append(self_attr)

                # compute the neighbor edge attr
                node1_nb_nodes = self.dataset.sample_nb_nodes_with_similar_score(node1, edge_score,
                                                                                 size=self.nb_walk_size)
                node2_nb_nodes = self.dataset.sample_nb_nodes_with_similar_score(node2, edge_score,
                                                                                 size=self.nb_walk_size)

                nb_edge_attr = torch.zeros(edges_attr[0].shape)
                each_edge_nb = []
                for nb_node in node1_nb_nodes:
                    each_edge_nb.append((node1, nb_node))
                    nb_edge_attr += torch.Tensor(self.dataset.get_combine_attr_from_nodes(node1, nb_node))
                for nb_node in node2_nb_nodes:
                    each_edge_nb.append((node2, nb_node))
                    nb_edge_attr += torch.Tensor(self.dataset.get_combine_attr_from_nodes(node2, nb_node))

                if self.config['type'] == 'mean':
                    nb_edge_attr /= (len(node1_nb_nodes) + len(node2_nb_nodes))

                edges_nb_attr.append(nb_edge_attr)

                # store the sampling for each edge
                self.edges_and_nb_dict[edge] = each_edge_nb

        else:
            # not the first aggregation layer, get the encoded edge from previous encode layer
            nb_edges = []
            for edge in edges:
                # compute target_edge attr
                edge_score = g.edges[edge]['score']
                node1 = edge[0]
                node2 = edge[1]

                # compute the neighbor edge attr
                node1_nb_nodes = self.dataset.sample_nb_nodes_with_similar_score(node1, edge_score,
                                                                                 size=self.nb_walk_size)
                node2_nb_nodes = self.dataset.sample_nb_nodes_with_similar_score(node2, edge_score,
                                                                                 size=self.nb_walk_size)
                each_edge_nb = []
                for nb_node in node1_nb_nodes:
                    nb_edges.append((node1, nb_node))
                    each_edge_nb.append((node1, nb_node))
                for nb_node in node2_nb_nodes:
                    nb_edges.append((node2, nb_node))
                    each_edge_nb.append((node2, nb_node))

                self.edges_and_nb_dict[edge] = each_edge_nb

            self.encoder(edges + nb_edges)  # calculate all necessary edges encoding from last encoding

            for self_edge in edges:
                edges_attr.append(self.encoder.encoded_edges[self_edge])

                nb_edge_attr = torch.zeros(edges_attr[0].shape)
                self_edge_nb = self.edges_and_nb_dict[self_edge]
                for nb_edge in self_edge_nb:
                    nb_edge_attr += self.encoder.encoded_edges[nb_edge]
                if self.config['type'] == 'mean':
                    nb_edge_attr /= (len(self_edge_nb))
                edges_nb_attr.append(nb_edge_attr)

        return torch.stack(edges_attr), torch.stack(edges_nb_attr)
