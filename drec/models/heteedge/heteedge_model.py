import torch
import torch.nn as nn
import torch.nn.functional as F


class MovielensSuperviseGraphSageModel(nn.Module):
    def __init__(self, num_classes, encoder):
        super(MovielensSuperviseGraphSageModel, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.Tensor(encoder.embed_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, edges):
        embeds = self.encoder(edges)
        scores = embeds.mm(self.weight)
        return scores
