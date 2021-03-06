import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 简单的三层全连接神经网络
class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


    def get_name(self):
        return self.__class__.__name__

# 添加激活函数
class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def get_name(self):
        return self.__class__.__name__

# 添加批标准化(加快收敛速度)
class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def get_name(self):
        return self.__class__.__name__

# 超参数
batch_size = 64
learning_rate = 1e-2
num_epoches = 5

# 数据预处理
data_tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)

# 获取训练数据集
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=data_tf, download=True
# )

# 从文件中读出训练数据
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf
)

# 从文件中读出测试数据
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=data_tf
)

# 构造迭代器DataLoader，每次会自动读出batch_size量的数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义模型，损失函数，优化器
model = [simpleNet(28 * 28, 300, 100, 10), Activation_Net(28 * 28, 300, 100, 10), Batch_Net(28 * 28, 300, 100, 10)]
# model = Activation_Net
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练过程
def train(net, train_data, valid_data, num_epoch, optimizer, criterion):

    trainAcc = []
    EPOCH = []
    # validAcc = []
    count = 0
    print(net)
    length = len(train_data)
    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for iter, data in enumerate(train_data):
            im, label = data
            im = im.view(im.size()[0], -1)
            im = Variable(im) # 转化成可以进行训练的张量
            # print(im.shape)
            label = Variable(label)
            output = net(im)    # 执行一次前向传播，得到网络的输出
            loss = criterion(output, label) # 计算网络的输出和真实值的损失
            # ------------------- 优化过程
            optimizer.zero_grad()   # 把梯度变为0
            loss.backward() # 执行一次反向传播
            optimizer.step()    # 进行一次优化
            # -------------------
            _, pred_label = torch.max(output.data, 1)
            train_loss += loss.data
            temp_loss = loss.data
            # train_acc += torch.sum(pred_label == label.data)
            train_acc += accuracy_score(label.data, pred_label) * label.size(0) # 累计计算预测的准确率

            # temp_acc = (torch.sum(pred_label == label.data)) / label.size(0)
            temp_acc = accuracy_score(label.data, pred_label)   # 计算预测的准确率
            count += 1
            trainAcc.append(temp_acc)
            EPOCH.append(count)
            if iter % 300 == 0 and iter > 0:
                # count += 1
                # trainAcc.append(temp_acc)
                # EPOCH.append(count)
                print('Epoch {}/{},Iter {}/{} Loss: {:.4f},ACC:{:.4f}' \
                      .format(epoch, num_epoches - 1, iter, length, temp_loss, temp_acc))
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net.eval()
            with torch.no_grad():
                for iter, data in enumerate(valid_data):
                    im, label = data
                    im = im.view(im.size()[0], -1)
                    im = Variable(im)
                    label = Variable(label)
                    output = net(im)
                    _, pred_label = torch.max(output.data, 1)
                    loss = criterion(output, label)
                    valid_loss += loss.data
                    # valid_acc += torch.sum(pred_label == label.data)
                    valid_acc += accuracy_score(label.data, pred_label) * label.size(0)
            print('Epoch {}/{},complete! train_loss: {:.4f},train_acc:{:.4f}' \
                  .format(epoch, num_epoches - 1, train_loss, train_acc / 60000),
                  'valid_loss: {:.4f},valid_acc:{:.4f}'.format(valid_loss, valid_acc / 10000)
                  )
    return EPOCH, trainAcc
for m in model:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print("the {} start traing...".format(m.get_name()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(m.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
    x, y = train(m, train_loader, test_loader, num_epoches, optimizer, criterion)
    ax.plot(x, y)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training loss')
    ax.set_title(m.get_name())
    plt.show()
    print("the {} complete traing...".format(m.get_name()))
