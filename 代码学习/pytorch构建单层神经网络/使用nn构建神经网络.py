import torch
import torch.nn as nn


N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in,requires_grad=True)
y=torch.randn(N,D_out,requires_grad=True)

# Sequential 一个时序容器。Modules会以他们传入的顺序被添加到容器中。
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),  # w1 * x + b1
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)

# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)

loss_fn=nn.MSELoss(reduction='sum') # reduction属性：设置对输出的计算方法

# pytorch优化器
# Adam 比较好的初始learning rate是1e-4 ~ 1e-3
# learning_rate=1e-4
# optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

learning_rate=1e-6
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for i in range(500):
    y_pred=model(x) # model.forward()
    loss=loss_fn(y_pred,y)
    print(i, loss.item())

    # 手动进行参数更新：
    # model.zero_grad()
    # loss.backward()
    # with torch.no_grad():
    #     # 所有的参数都在model.parameters
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad


    # 使用pytorch优化器进行参数更新:
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
