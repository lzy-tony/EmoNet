import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.embed_dim = config.emb_dim
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config.emb_dim * config.max_len, config.hidden_size1),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size1, config.hidden_size2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size2, config.class_num),
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit
