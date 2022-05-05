from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

import text_cnn.model
import text_cnn.config
import text_cnn.preprocess
import lstm.model
import lstm.config
import lstm.preprocess

import training


def load(m: str):
    if m == 'cnn':
        config = text_cnn.config.Config()
        train_dataloader, test_dataloader = text_cnn.preprocess.load_data()
        model = text_cnn.model.TextCNN(config=config)
        return config, train_dataloader, test_dataloader, model
    elif m == 'lstm':
        config = lstm.config.Config()
        train_dataloader, test_dataloader = lstm.preprocess.load_data()
        model = lstm.model.LSTM(config=config)
        return config, train_dataloader, test_dataloader, model


def main(m: str):
    config, train_dataloader, test_dataloader, model = load(m)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    training.main(train_dataloader, test_dataloader, model, device, loss_fn, optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        choices=['cnn', 'lstm'], help='choose model')
    args = parser.parse_args()
    main(args.model)
