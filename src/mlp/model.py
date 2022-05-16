import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.embed_dim = config.emb_dim
        self.flatten = nn.Flatten()

        modules = [nn.Linear(config.emb_dim * config.max_len, config.hidden_size),
                   nn.ReLU(),
                   nn.Dropout(config.dropout_rate)]
        for i in range(config.hidden_num - 1):
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(config.dropout_rate))
        modules.append(nn.Linear(config.hidden_size, config.class_num))
        self.linear_relu_stack = nn.Sequential(*modules)

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit
