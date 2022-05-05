from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

import text_cnn.model
import text_cnn.config
import text_cnn.preprocess
import training


def main():
    config = text_cnn.config.Config()
    train_dataloader, test_dataloader = text_cnn.preprocess.load_data()
    model = text_cnn.model.TextCNN(config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    training.main(train_dataloader, test_dataloader, model, device, loss_fn, optimizer)


if __name__ == '__main__':
    # parser = ArgumentParser()
    main()
