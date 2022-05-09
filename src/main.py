from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

import text_cnn.model, text_cnn.config
import lstm.model, lstm.config
import mlp.model, mlp.config

import training
import preprocess


def load(m: str):
    train_dataloader, test_dataloader = preprocess.load_data()
    if m == 'cnn':
        config = text_cnn.config.Config()
        model = text_cnn.model.TextCNN(config=config)
        return config, train_dataloader, test_dataloader, model
    elif m == 'lstm':
        config = lstm.config.Config()
        model = lstm.model.LSTM(config=config)
        return config, train_dataloader, test_dataloader, model
    elif m == 'mlp':
        config = mlp.config.Config()
        model = mlp.model.MLP(config=config)
        return config, train_dataloader, test_dataloader, model


def main(m: str):
    config, train_dataloader, test_dataloader, model = load(m)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    training.main(train_dataloader, test_dataloader, model, device, loss_fn, optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'mlp'], help='choose model')
    args = parser.parse_args()
    main(args.model)
