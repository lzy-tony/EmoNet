from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim

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


def load_list(m: str, c: str):
    train_dataloader, test_dataloader = preprocess.load_data()
    config_list = []
    label_list = []
    if m == 'cnn':
        if c == 'kernel_num':
            config_list = [text_cnn.config.KNConfig1, text_cnn.config.KNConfig2,
                           text_cnn.config.KNConfig3]
            label_list = [config.kernel_num for config in config_list]
        elif c == 'kernel_size':
            config_list = [text_cnn.config.KSConfig1, text_cnn.config.KSConfig2,
                           text_cnn.config.KSConfig3, text_cnn.config.KSConfig4]
            label_list = [config.kernels for config in config_list]
        elif c == 'dropout':
            config_list = [text_cnn.config.DConfig1, text_cnn.config.DConfig2,
                           text_cnn.config.DConfig3, text_cnn.config.DConfig4]
            label_list = [config.dropout_rate for config in config_list]
        model_list = [text_cnn.model.TextCNN(config=config) for config in config_list]
        return config_list, train_dataloader, test_dataloader, model_list, label_list
    elif m == 'lstm':
        if c == 'dropout':
            config_list = [lstm.config.DConfig1, lstm.config.DConfig2,
                           lstm.config.DConfig3, lstm.config.DConfig4]
            label_list = [config.dropout_rate for config in config_list]
        model_list = [lstm.model.LSTM(config=config) for config in config_list]
        return config_list, train_dataloader, test_dataloader, model_list, label_list
    else:
        if c == 'hidden_size':
            config_list = [mlp.config.HSConfig1, mlp.config.HSConfig2,
                           mlp.config.HSConfig3, mlp.config.HSConfig4,
                           mlp.config.HSConfig5]
            label_list = [config.hidden_size for config in config_list]
        elif c == 'hidden_num':
            config_list = [mlp.config.HNConfig1, mlp.config.HNConfig2,
                           mlp.config.HNConfig3]
            label_list = [config.hidden_num for config in config_list]
        elif c == 'dropout':
            config_list = [mlp.config.DConfig1, mlp.config.DConfig2,
                           mlp.config.DConfig3, mlp.config.DConfig4]
            label_list = [config.dropout_rate for config in config_list]
        model_list = [mlp.model.MLP(config=config) for config in config_list]
        return config_list, train_dataloader, test_dataloader, model_list, label_list


def validate(m: str):
    validate_dataloader = preprocess.load_validator()
    path = "..\\models\\"
    if m == 'cnn':
        path += "Text-CNN.pth"
        config = text_cnn.config.Config()
        model = text_cnn.model.TextCNN(config=config)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return validate_dataloader, model
    elif m == 'lstm':
        path += "LSTM.pth"
        config = lstm.config.Config()
        model = lstm.model.LSTM(config=config)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return validate_dataloader, model
    elif m == 'mlp':
        path += "MLP.pth"
        config = mlp.config.Config()
        model = mlp.model.MLP(config=config)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return validate_dataloader, model


def main(m: str, c: str, v: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    if v == 'true':
        validate_dataloader, model = validate(m)
        training.validate(validate_dataloader, model, device, loss_fn, m)
    elif c == 'false':
        config, train_dataloader, test_dataloader, model = load(m)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        training.main(train_dataloader, test_dataloader, model, device, loss_fn, optimizer, config)
    else:
        config_list, train_dataloader, test_dataloader, model_list, label_list = load_list(m, c)
        optimizer_list = [optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                          for config, model in zip(config_list, model_list)]
        training.cmp_params(train_dataloader, test_dataloader,
                            model_list, device, loss_fn, optimizer_list, config_list, label_list, c)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'mlp'], help='choose model')
    parser.add_argument('-c', '--compare', type=str, default='false',
                        choices=['false', 'kernel_num', 'kernel_size', 'dropout',
                                 'hidden_num', 'hidden_size', 'weight_decay'], help='compare params')
    parser.add_argument('-v', '--validate', type=str, default='false',
                        choices=['false', 'true'], help='validate model')
    args = parser.parse_args()
    main(args.model, args.compare, args.validate)
