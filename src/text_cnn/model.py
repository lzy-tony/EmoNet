import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        # params
        D = config.emb_dim
        Cin = 1
        Cout = config.kernel_num
        Ks = config.kernels

        # conv layers
        self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (K, D)) for K in Ks])

        # fc layer
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(len(Ks)*Cout, config.class_num)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # input: (N, W, D)
        x = x.unsqueeze(1)  # unsqueezed: (N, Cin, W, D)
        conv_x = [F.relu(conv(x)).squeeze(-1)
                  for conv in self.convs]  # conv: (N, Cout, W) * len(Ks)
        pool_x = [F.max_pool1d(w, w.size(2)).squeeze(2)
                  for w in conv_x]  # pooled: (N, Cout) * len(Ks)
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        # logit = self.softmax(fc_x)
        return logit
