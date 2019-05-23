import torch
import torch.nn as nn


class FeedforwardModel(nn.Module):

    def __init__(self, config):
        super(FeedforwardModel, self).__init__()
        self.my_layers = nn.ModuleList()
        self.config = config
        self._process_config()

    def _process_config(self):
        layers = self.config['layers']
        for idx, layer_tuple in enumerate(layers):
            type = layer_tuple[0]
            if type is 'linear':
                self.my_layers.append(nn.Linear(layer_tuple[1], layer_tuple[2]))
            if type is 'relu':
                self.my_layers.append(nn.ReLU())
            if type is 'dropout':
                self.my_layers.append(nn.Dropout(p=layer_tuple[1]))
            if type is 'softmax':
                self.my_layers.append(nn.Softmax(dim=0))

    def forward(self, x):
        out = x
        if torch.cuda.is_available():
            out = x.cuda()

        for m in self.my_layers:
            out = m(out)

        return out
