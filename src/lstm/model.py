import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.embed_dim = config.emb_dim
        self.max_len = config.max_len
        self.lstm = nn.LSTM(input_size=config.emb_dim,
                            hidden_size=config.hidden_layer_num,
                            num_layers=2,
                            dropout=config.dropout_rate,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.hidden_layer_num * config.max_len, config.class_num)

    def forward(self, x):
        lstm_out, h = self.lstm(x)
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)
        fc_x = self.dropout(lstm_out)
        pred = self.fc(fc_x)
        return pred

