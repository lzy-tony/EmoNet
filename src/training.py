import torch
import pylab as mpl
import hiddenlayer as hl
from matplotlib import pyplot as plt
from ignite.metrics import Precision, Recall

import text_cnn.model, text_cnn.config


def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # run model
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print result
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")


def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    precision = Precision(average=True)
    recall = Recall(average=True)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            precision.update((pred, y))
            recall.update((pred, y))
    test_loss /= num_batches
    correct /= size
    p = precision.compute()
    r = recall.compute()
    f = p * r * 2 / (p + r + 1e-20)

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, "
          f"Precision: {p}, Recall: {r}, F: {f} \n")
    return test_loss, 100 * correct, f


def main(train_dataloader, test_dataloader, model, device, loss_fn, optimizer, config):
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    epoch = 20
    acc_list = [0]
    loss_list = []
    f_list = []
    for i in range(epoch):
        train(train_dataloader, model, device, loss_fn, optimizer)
        loss, accuracy, f = test(test_dataloader, model, device, loss_fn)
        acc_list.append(accuracy)
        loss_list.append(loss)
        f_list.append(f)
    torch.save(model.state_dict(), '..\\models\\' + config.name + ".pth")
    plt.title(f"{config.name} 测试集正确率")
    plt.plot(range(epoch + 1), acc_list)
    plt.savefig("..\\res\\" + config.name + "_accuracy.svg")
    plt.close()
    plt.title(f"{config.name} 测试集 loss")
    plt.plot(range(1, epoch + 1), loss_list)
    plt.savefig("..\\res\\" + config.name + "_loss.svg")
    plt.close()
    plt.title(f"{config.name} 测试集 f-score")
    plt.plot(range(1, epoch + 1), f_list)
    plt.savefig("..\\res\\" + config.name + "_f.svg")
    plt.close()


def cmp_params(train_dataloader, test_dataloader, model_list,
               device, loss_fn, optimizer_list, config_list, label_list, c):
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    for model, optimizer, config, label in zip(model_list, optimizer_list, config_list, label_list):
        epoch = 20
        acc_list = [0]
        for i in range(epoch):
            print(f"running epoch {i} in label {label}")
            train(train_dataloader, model, device, loss_fn, optimizer)
            loss, accuracy, f = test(test_dataloader, model, device, loss_fn)
            acc_list.append(accuracy)
        plt.plot(range(epoch + 1), acc_list, label=label)
    plt.legend()
    plt.title(f"{config_list[0].name} 测试集正确率 - 参数 {c} 比较")
    plt.savefig("..\\res\\" + config_list[0].name + "_" + c + ".svg")
    plt.close()


def validate(validate_dataloader, model, device, loss_fn, m):
    loss, accuracy, f = test(validate_dataloader, model, device, loss_fn)
    print(f"on validation set, loss = {loss}, accuracy = {accuracy}, f = {f}")
    tensor = torch.zeros(1, 100, 50)
    hl_graph = hl.build_graph(model, tensor)
    hl_graph.theme = hl.graph.THEMES["blue"].copy()
    hl_graph.save("..\\models\\" + m + "_graph.png")