import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteEdgeEncoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    List of keyword args:
     - layer: a layer number to keep track (since there is many recursive calling of encoder and aggregator)
     - dropout: enable dropout layer with dropout value
     - aggre_type: the aggregation function, can be "mean", "maxpool", "meanpool", "lstm"
    """

    def __init__(self,
                 feature_dim,
                 embed_dim,
                 aggregator,
                 **kwargs):
        super(HeteEdgeEncoder, self).__init__()
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.aggregator = aggregator
        self.layer = kwargs.get('layer', 1)
        self.dropout_value = kwargs.get('dropout', -1)
        self.dropout = None
        self.aggre_type = kwargs.get('aggre_type', "maxpool")
        self._configure()

        self.encoded_edges = {}
        self.weight = nn.Parameter(torch.Tensor(2 * self.feat_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def _configure(self):
        if self.dropout_value > 0 and self.dropout_value < 1:
            self.dropout = nn.Dropout(p=self.dropout_value)


    def forward(self, edges):

        # first look for saved encoded_edge
        none_edges = []
        for edge in edges:
            if self.encoded_edges.get(edge, None) is None:
                none_edges.append(edge)

        # if exist none value, aggre them
        if len(none_edges) > 0:
            edges_attr, edges_nb_attr = self.aggregator(none_edges)
            combine_attrs = torch.cat((edges_attr, edges_nb_attr), dim=1)
            activate = F.relu(combine_attrs.mm(self.weight))

            for index, edge in enumerate(none_edges):
                self.encoded_edges[edge] = activate[index, :]

        edge_encoded_attr = torch.stack([self.encoded_edges[edge] for edge in edges])
        if self.dropout is not None:
            edge_encoded_attr = self.dropout(edge_encoded_attr)
        return edge_encoded_attr
