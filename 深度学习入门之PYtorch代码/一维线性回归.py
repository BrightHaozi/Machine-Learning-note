import torch
from torch.nn import Linear
from torch import nn
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt

# 获得训练数据
x = sorted([random.randint(-1000, 1000) * 0.01 for i in range(10)])
y = sorted([random.randint(-1000, 1000) * 0.01 for i in range(10)])
x_train = [[i] for i in x]
y_train = [[i] for i in y]
x_train = np.array(x_train,dtype=np.float32)
y_train = np.array(y_train,dtype=np.float32)

print(x_train, '\n', y_train)

# 画图
# plt.scatter(x_train, y_train)
# plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
plt.scatter(x_train.numpy(), y_train.numpy())
plt.show()
# x_train = x_train.float()
# y_train = x_train.float()
# x_train = torch.tensor(x_train, requires_grad=True)
# y_train = torch.tensor(y_train, requires_grad=True)
# print(x_train, '\n', y_train)

# plt.scatter(x_train.detach().numpy(), y_train.detach().numpy())
# plt.show()

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

# 定义损失函数和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    out = model(inputs)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss : {:.6f}'.format(epoch + 1, num_epochs, loss.item()))

# 将模型变成测试模式
model.eval()

predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()