import math

import torch
import torch.nn as  nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np


# def getAggregator(input_dim, output_dim, aggre_type, sampler, *arg, **kwargs):
#     if aggre_type == 'maxpool':
#         return MultiSageNodeMaxPoolAggregator(input_dim, output_dim, sampler, arg, kwargs)


class HeteGraphRecNodeAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HeteGraphRecNodeAggregator, self).__init__()

    def _pooling_operation(self, nbs_attrs):
        raise Exception('imlentation by subclass')

    def _aggregate_nodes_recurrent(self, nodes):
        nbs_sampling = self.sampler.sample_neighborhood(nodes)

        # get unique nodes
        key_nodes = set(nbs_sampling.keys())
        nb_nodes = set([nbn for nbs in nbs_sampling.values() for nb_nodes in nbs for nbn in nb_nodes])
        unique_nodes_sorted = sorted(list(set.union(key_nodes, nb_nodes)))
        nodes_embeds_sorted = self.pre_aggregator(unique_nodes_sorted)

        nodes_attr_dict = {node: nodes_embeds_sorted[unique_nodes_sorted.index(node), :] for idx, node in
                           enumerate(nodes)}
        nodes_attr_tensor = torch.stack(list(nodes_attr_dict.values()))

        nodes_nbs_attr_dict = self._cal_nbs_aggregation_recurrent(nbs_sampling,
                                                                  unique_nodes_sorted,
                                                                  nodes_embeds_sorted)
        nodes_nbs_attr_tensor = torch.stack(list(nodes_nbs_attr_dict.values()))

        output = self._compute_output(nodes_attr_tensor, nodes_nbs_attr_tensor)
        return output

    def _cal_nbs_aggregation_recurrent(self, nbs_sampling, unique_nodes_sorted, nodes_embeds_sorted):
        nodes_nbs_attr_dict = {}
        for node, nbs_list in nbs_sampling.items():
            try:
                nbs_attr = self.sampler.get_nbs_attr_from_embeddings(node,
                                                                     nbs_list,
                                                                     unique_nodes_sorted,
                                                                     nodes_embeds_sorted)
                nbs_attr = torch.Tensor(nbs_attr)

                # perform pooling operation on node's neighbor
                nodes_nbs_attr_dict[node] = self._pooling_operation(nbs_attr)
            except Exception as e:
                print(e)
        return nodes_nbs_attr_dict

    def _aggregate_nodes_first_time(self, nodes):
        nodes_attr_dict = self.sampler.ds.get_nodes_attr(nodes)
        nodes_attr_tensor = torch.Tensor(np.stack(nodes_attr_dict.values()))

        nbs_sampling = self.sampler.sample_neighborhood(nodes)
        nodes_nbs_attr_dict = self._cal_nbs_aggregation_first_time(nbs_sampling)
        nodes_nbs_attr_tensor = torch.stack(list(nodes_nbs_attr_dict.values()))

        output = self._compute_output(nodes_attr_tensor, nodes_nbs_attr_tensor)
        return output

    def _cal_nbs_aggregation_first_time(self, nbs_sampling):
        nodes_nbs_attr_dict = {}
        for node, nbs_list in nbs_sampling.items():
            nbs_attr = self.sampler.get_nbs_attr(node, nbs_list)
            nbs_attr = torch.Tensor(nbs_attr)
            nodes_nbs_attr_dict[node] = self._pooling_operation(nbs_attr)

        return nodes_nbs_attr_dict

    def _compute_output(self, node_attr_tensor, node_nbs_attr_tensor):
        from_self = node_attr_tensor.matmul(self.self_weight)
        from_nb = node_nbs_attr_tensor.matmul(self.nb_weight)

        if self.combine_type == 'concat':
            output = torch.cat((from_self, from_nb), dim=1)
        if self.combine_type == 'mean':
            output = torch.add(from_self, from_nb)

        output = torch.add(output, self.bias)
        output = self.activation(output)
        # output = output / torch.norm(output, 2)
        return output


class HeteGraphRecNodeMaxPoolAggregator(HeteGraphRecNodeAggregator):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    kwarg values:
    - layer: a layer number to keep track (since there is many recursive calling of encoder and aggregator)
    """

    def __init__(self, input_dim, output_dim, sampler, *args, **kwargs):
        super(HeteGraphRecNodeMaxPoolAggregator, self).__init__(args, kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sampler = sampler
        self.pre_aggregator = kwargs.get('pre_aggregator', None)
        self.combine_type = kwargs.get('combine_type', "concat")
        self.dropout = kwargs.get('dropout', 0.5)  # todo: use this dropout
        self.pool_input_dim = kwargs.get('pool_input_dim', 32)
        self.pool_hidden_dim = kwargs.get('pool_hidden_dim', 512)
        self.pool_dropout = kwargs.get('pool_dropout', 0.5)
        self.activation = kwargs.get('activation', nn.ReLU())

        self.pooling_mlp = nn.Sequential(nn.Linear(self.pool_input_dim, self.pool_hidden_dim, bias=True),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.pool_dropout))
        self._init_weights_bias()

    def _init_weights_bias(self):
        self.nb_weight = nn.Parameter(torch.Tensor(self.pool_hidden_dim, self.output_dim))
        self.self_weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        init.xavier_uniform_(self.nb_weight)
        init.xavier_uniform_(self.self_weight)

        size = self.output_dim
        self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        stdv = 1. / math.sqrt(size)
        if self.combine_type == 'concat':
            size = self.output_dim * 2
            self.bias = nn.Parameter(torch.Tensor(size))
            stdv = 1. / math.sqrt(size)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        if self.pre_aggregator:
            output = self._aggregate_nodes_recurrent(nodes)
        else:
            output = self._aggregate_nodes_first_time(nodes)

        return output

    def _pooling_operation(self, nbs_attrs):
        max_pool_nbs_attr = self.pooling_mlp(nbs_attrs)
        return torch.max(max_pool_nbs_attr, dim=0)[0]


class HeteGraphRecNodeMeanPoolAggregator(HeteGraphRecNodeAggregator):
    def __init__(self, input_dim, output_dim, sampler, *args, **kwargs):
        super(HeteGraphRecNodeMeanPoolAggregator, self).__init__(args, kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sampler = sampler
        self.pre_aggregator = kwargs.get('pre_aggregator', None)
        self.combine_type = kwargs.get('combine_type', "concat")
        self.dropout = kwargs.get('dropout', 0.5)  # todo: use this dropout
        self.pool_input_dim = kwargs.get('pool_input_dim', 30)
        self.pool_hidden_dim = kwargs.get('pool_hidden_dim', 512)
        self.pool_dropout = kwargs.get('pool_dropout', 0.5)
        self.activation = kwargs.get('activation', nn.ReLU())

        self.pooling_mlp = nn.Sequential(nn.Linear(self.pool_input_dim, self.pool_hidden_dim, bias=True),
                                             nn.ReLU(),
                                             nn.Dropout(p=self.pool_dropout))
        self._init_weights_bias()

    def _init_weights_bias(self):
        self.nb_weight = nn.Parameter(torch.Tensor(self.pool_hidden_dim, self.output_dim))
        self.self_weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        init.xavier_uniform_(self.nb_weight)
        init.xavier_uniform_(self.self_weight)

        size = self.output_dim
        self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        stdv = 1. / math.sqrt(size)
        if self.combine_type == 'concat':
            size = self.output_dim * 2
            self.bias = nn.Parameter(torch.Tensor(size))
            stdv = 1. / math.sqrt(size)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        if self.pre_aggregator:
            output = self._aggregate_nodes_recurrent(nodes)
        else:
            output = self._aggregate_nodes_first_time(nodes)

        return output

    def _pooling_operation(self, nbs_attrs):
        mean_pool_nbs_attr = self.pooling_mlp(nbs_attrs)
        return torch.mean(mean_pool_nbs_attr, dim=0)
