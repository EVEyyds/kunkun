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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取 CSV 文件
    train_data = pd.read_csv('./train.csv')
    test_data= pd.read_csv('./test.csv')

    x_train,y_train=preprocessing(train_data)
    print("训练集预处理完成")


    x_test,y_test=preprocessing(test_data)
    print("测试集预处理完成")


    dl_train = DataLoader(TensorDataset(x_train, y_train),
                          shuffle=True, batch_size=100, num_workers=4)
    dl_valid = DataLoader(TensorDataset(x_test, y_test),
                          shuffle=False, batch_size=100, num_workers=4)
    net = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0008)

    metric_name = "accuracy"

    epochs = 10
    log_step_freq = 30
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])

    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("==========" * 8 + "%s" % nowtime)
    for epoch in range(1, epochs + 1):
        # 1，训练循环-------------------------------------------------
        net.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            features=features.to(device)
            labels=labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 正向传播求损失

            predictions = net(features)

            loss = loss_func(predictions, labels)
            # print("loss:",loss)
            metric = metric_func(predictions, labels)
            # 反向传播求梯度
            loss.backward()
            optimizer.step()
            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(
                    ("[step = %d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step))

        # 2，验证循环-------------------------------------------------
        net.eval().to(device)
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            predictions = net(features)
            val_loss = loss_func(predictions, labels)
            val_metric = val_metric_func(predictions, labels)
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()


        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info
        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + " = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')

    # plot_metric(dfhistory, "loss")
    # plot_metric(dfhistory, "accuracy")

    # 保存模型参数
    torch.save(net.state_dict(), "./net_parameter.pkl")

