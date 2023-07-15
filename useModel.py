import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize


#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

def preprocessing(dfdata):
    x = torch.tensor(np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(1, 48, 48)/255.0 for pixel in dfdata['pixels']])).float()
    labels = dfdata['emotion'].values.astype(np.int64)
    y=torch.tensor(np.eye(7)[labels]).float()
    return x,y

def preprocessing2(dfdata):
    x = torch.tensor(np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(1, 48, 48) / 255.0 for pixel in dfdata])).float()
    return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.softmax(self.fc2(x))
        return x

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

def metric_func(y_pred,y_true):
    y_pred=np.array([i.detach().numpy() for i in y_pred])
    y_true=np.array([i.detach().numpy() for i in y_true])
    y_pred_labels=np.argmax(y_pred, axis=1)
    y_true_labels = np.squeeze(y_true)

    y_pred_bin = label_binarize(y_pred_labels, classes=np.arange(y_pred.shape[1]))

    # print(y_true_labels)
    # print(y_pred_bin)

    acc = accuracy_score(y_true_labels, y_pred_bin)

    return acc

def val_metric_func(y_pred,y_true):
    y_pred = np.array([i.detach().numpy() for i in y_pred])
    y_true = np.array([i.detach().numpy() for i in y_true])
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.squeeze(y_true)

    y_pred_bin = label_binarize(y_pred_labels, classes=np.arange(y_pred.shape[1]))

    acc = accuracy_score(y_true_labels, y_pred_bin)
    return acc

if __name__ == '__main__':
    dftest_raw=pd.read_csv("./val.csv")
    # print(dftest_raw.shape)
    # print(type(dftest_raw["pixels"][0]))
    x_test,y_test=preprocessing(dftest_raw)
    net_clone=CNN()
    net_clone.load_state_dict(torch.load("./net_parameter63.pkl"))
    y_pred_probs=net_clone(x_test)
    # print(y_pred_probs)
    metric=metric_func(y_pred_probs,y_test)
    print("准确率:",metric)





