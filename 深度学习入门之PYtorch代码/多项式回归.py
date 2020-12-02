import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import random

import matplotlib.pyplot as plt
import numpy as np

# 定义最高次项的指数
n = 3

# 为线性相乘做准备，将x中的每一个元素排列成[x^1, x^2, ..., x^n]
# 因为有多个x，故返回的是一个4*4的二维张量
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

# 不加入x^0的原因是，nn.Linear默认自带bias属性（可取消）

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

# 定义真实的函数

def f(x):
    return x.mm(W_target) + b_target[0]

# 生成训练集。
def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)

class poly_model(nn.Module):
    def __init__(self, n):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(n, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

# 实例化模型
model = poly_model(n)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#开始模型训练
epoch = 0
while True:
    # 获得数据
    batch_x, batch_y = get_batch()
    # 前向传播
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss=loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch += 1
    if print_loss < 1e-3:
        break




# 定义函数输出形式
def func_format(weight, bias, n):
    func = ''
    for i in range(n, 0, -1):
        func += ' {:.2f} * x^{} +'.format(weight[i - 1], i)
    return 'y =' + func + ' {:.2f}'.format(bias[0])


predict_weight = model.poly.weight.data.numpy().flatten()
predict_bias = model.poly.bias.data.numpy().flatten()
print('predicted function :', func_format(predict_weight, predict_bias, n))
real_W = W_target.numpy().flatten()
real_b = b_target.numpy().flatten()
print('real      function :', func_format(real_W, real_b, n))