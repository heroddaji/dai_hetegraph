import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import drec.utils as utils


class HeteGraphRecModel(nn.Module):
    def __init__(self, input_dim, output_dim, aggregator, sampler, *args, **kwargs):
        super(HeteGraphRecModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.sampler = sampler
        self.combine_type = kwargs.get('combine_type', None)
        self._init_weights_bias()

    def _init_weights_bias(self):
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if self.combine_type == 'concat':
            self.weight = nn.Parameter(torch.Tensor(self.input_dim * 2, self.output_dim))

        self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, items):
        raise Exception('Implement in subclass')

    def get_negative_nodes(self, nodes, size=200, types={'user': 'item', 'item': 'user'}):
        unique_nodes = utils.get_unique_nodes_from_nodes(nodes)
        neg_nodes_dict = self.sampler.sample_negative_nodes(unique_nodes, size=size, types=types)
        return neg_nodes_dict

    def max_margin_ranking_loss(self,
                                edges,
                                neg_nodes_dict,
                                unique_nodes,
                                unique_nodes_embedding,
                                graph,
                                margin=0.1):
        # loss function from this paper https://arxiv.org/pdf/1806.01973.pdf
        loss = 0
        for edge in edges:
            node1 = edge[0]
            node1_label = graph.nodes[node1]['label']
            node2 = edge[1]
            node2_label = graph.nodes[node2]['label']

            # vectorize the operation
            node1_embed = unique_nodes_embedding[unique_nodes.index(node1)]
            node1_embeds = node1_embed.repeat(len(neg_nodes_dict[node1_label]), 1)

            node2_embed = unique_nodes_embedding[unique_nodes.index(node2)]
            node2_embeds = node2_embed.repeat(len(neg_nodes_dict[node1_label]), 1)

            neg_nodes_embed = torch.stack(
                [unique_nodes_embedding[unique_nodes.index(i)] for i in neg_nodes_dict[node1_label]])
            node_pair_loss = torch.mean(torch.max(torch.zeros(node1_embeds.shape[0]),
                                                  node1_embeds.matmul(neg_nodes_embed.t()) - node1_embeds.matmul(
                                                      node2_embeds.t()) + margin))

            loss += node_pair_loss
        return loss


class HeteGraphRecEdgeRegression(HeteGraphRecModel):
    def __init__(self, input_dim, output_dim, aggregator, sampler, *args, **kwargs):
        super(HeteGraphRecEdgeRegression, self).__init__(input_dim, output_dim, aggregator, sampler, *args, **kwargs)

    def forward(self, edges):
        unique_nodes = utils.get_unique_nodes_from_edges(edges)
        nodes_embeds = self.aggregator(unique_nodes)

        # combine node to become edge embedding
        edges_embedding = []
        unique_edges = []
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            node1_emb = nodes_embeds[unique_nodes.index(node1), :]
            node2_emb = nodes_embeds[unique_nodes.index(node2), :]
            if self.combine_type == 'concat':
                edges_embedding.append(torch.cat((node1_emb, node2_emb)))
            unique_edges.append((node1, node2))
        edges_embedding = torch.stack(edges_embedding)
        edges_embedding = edges_embedding.mm(self.weight) + self.bias
        return edges_embedding, unique_edges


class HeteGraphRecNodeEmbedding(HeteGraphRecModel):
    def __init__(self, input_dim, hidden_dim, output_dim, aggregator, sampler, *args, **kwargs):
        super(HeteGraphRecNodeEmbedding, self).__init__(input_dim, output_dim, aggregator, sampler, *args, **kwargs)
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self._init_weights_bias2()

    def _init_weights_bias2(self):
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.Tensor(self.hidden_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

        self.weight2 = nn.Parameter(torch.Tensor(self.hidden_dim, self.output_dim))
        init.xavier_uniform_(self.weight2)

    def forward(self, edges, negative_types):
        self.train_edges = edges

        self.unique_nodes = utils.get_unique_nodes_from_edges(edges)
        self.neg_nodes_dict = self.get_negative_nodes(self.unique_nodes, size=200,
                                                      types=negative_types)
        negative_nodes = [ne_node for k, v in self.neg_nodes_dict.items() for ne_node in v]

        self.unique_nodes = utils.get_unique_nodes_from_nodes(self.unique_nodes + negative_nodes)
        nodes_embeds = self.aggregator(self.unique_nodes)
        nodes_embeds = nodes_embeds.mm(self.weight) + self.bias
        self.unique_nodes_embedding = self.relu(nodes_embeds).mm(self.weight2)
        return self.unique_nodes_embedding, self.unique_nodes

    def loss(self, margin=0.1):
        return self.max_margin_ranking_loss(self.train_edges,
                                     self.neg_nodes_dict,
                                     self.unique_nodes,
                                     self.unique_nodes_embedding,
                                     self.sampler.ds.graph,
                                     margin=margin)
