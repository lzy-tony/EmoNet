from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim

import text_cnn.model
import text_cnn.preprocess
import training


def main():
    emb_dim = 50
    kernels = [2, 3, 4, 5]
    dropout_rate = 0.1
    class_num = 2
    train_dataloader, test_dataloader = text_cnn.preprocess.load_data()
    model = text_cnn.model.textCNN(embed_dim=emb_dim, class_num=class_num,
                                   dropout=dropout_rate, kernel_num=4,
                                   kernel_sizes=kernels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3)

    for i in range(10):
        training.train(train_dataloader, model, device, loss_fn, optimizer)


if __name__ == '__main__':
    # parser = ArgumentParser()
    main()