import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in,requires_grad=True)
y=torch.randn(N,D_out,requires_grad=True)

w1=torch.randn(D_in,H,requires_grad=True)
w2=torch.randn(H,D_out,requires_grad=True)

learning_rate=1e-6


for i in range(500):
    # forward propagration
    h=torch.mm(x,w1)
    h_relu=h.clamp(min=0)
    y_pred=torch.mm(h_relu,w2)

    # y_pred=x.mm(w1).clamp(min=0).mm(w2)

    loss=(y_pred-y).pow(2).sum()  # computation graph
    print("i={0}   loss={1}".format(i,loss.item()))

    loss.backward()
    with torch.no_grad(): # 节省内存，不会把w1,w2的gradient记录下来
        w1-=learning_rate*w1.grad # 这也是一个计算图
        w2-=learning_rate*w2.grad
        # w2=w2-learning_rate*w2.grad # 报错！

        # torch中每次计算完的梯度会自动进行累加，因此此处需要对上次计算出的梯度进行清零
        w1.grad.zero_()
        w2.grad.zero_()
