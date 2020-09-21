import numpy as np

# 使用numpy手动构建一个单层神经网络

#N 是输入数据组数，D_in是输入数据维度，H是hidden layor的神经元个数,D_out是输出数据维度
N,D_in,H,D_out=64,1000,100,10

#create training data randomly
x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)
w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)

learning_rate=1e-6

for t in range(500):
    # forward pass
    h=x.dot(w1) # h=w·x
    h_relu=np.maximum(0,h)   # relu activation
    y_pred=h_relu.dot(w2)

    # compute loss
    # mean square error
    loss=np.square(y_pred-y).sum()
    print(t,loss)

    # backward pass
    # compute the gradient
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.T.dot(grad_y_pred)

    grad_h_relu=grad_y_pred.dot(w2.T)
    grad_h=grad_h_relu.copy()
    grad_h[h<0]=0
    grad_w1=x.T.dot(grad_h)

    w1=w1-learning_rate*grad_w1
    w2=w2-learning_rate*grad_w2