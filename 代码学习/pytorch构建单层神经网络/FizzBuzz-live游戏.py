import torch
import torch.nn as nn
import numpy as np

#FizzBuzz是一个简单的小游戏。从1开始往上数数，遇到3的倍数说fizz，遇到3的倍数说fizz
#遇到5的倍数说buzz，遇到15的倍数说fizzbuzz。其他正常数数
#写一个简单程序决定要返回正常数值还是fizz,buzz or fizzbuzz

def fizz_buzz_encode(i):
    if i % 15==0:return 3
    elif i%5==0:return 2
    elif i%3==0:return 1
    else: return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]

def helper(i):
    #print(fizz_buzz_decode(i,fizz_buzz_encode(i)))
    return fizz_buzz_decode(i,fizz_buzz_encode(i))

# for i in range(1,16):
#     helper(i)

# 让神经网络学会玩这个游戏


# 定义模型的输入与输出
NUM_DIGITS = 10 # 表示一个数用多少个二进制位进行表示

# 将十进制转化为长度为10的二进制编码
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])

# 0-100作为testing data，>100作为training data

#生成训练数据
trX=torch.Tensor([binary_encode(i,NUM_DIGITS)for i in range(101,2**NUM_DIGITS)])
trY=torch.LongTensor([fizz_buzz_encode(i)for i in range(101,2**NUM_DIGITS)])

NUM_HIDDEN = 100
model=nn.Sequential(
    nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    nn.ReLU(),
    nn.Linear(NUM_HIDDEN,4) # 4 logits, after softmax, we get a probability distribution
)

loss_fn=torch.nn.CrossEntropyLoss() #分类问题常用交叉熵损失
# optimizer=torch.optim.SGD(model.parameters(),lr=0.05)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

BATCH_SIZE=128

for epoch in range(2000):
    for start in range(0,len(trX),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchX=trX[start:end]
        batchY=trY[start:end]

        y_pred=model(batchX)
        loss=loss_fn(y_pred,batchY)

        print("epoch:{}  loss:{}".format(epoch,loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # gradient descent

testX=torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,101)])
with torch.no_grad():
    testY=model(testX)
predictions=zip(range(1,101),testY.max(1)[1].data.tolist())
temp=zip(range(1,101),testY.max(1)[1].data.tolist()) #用于测试正确率
print([fizz_buzz_decode(i,x) for i,x in predictions])  #用于可视化结果

count=0
for i,x in temp:
    if fizz_buzz_decode(i,x)==helper(i):count+=1
#temp=[fizz_buzz_decode(i,x)==helper(i) for i,x in predictions]
# print(temp)
# count=temp.count(True)
print(count)
print("正确率：{}".format(count/100))