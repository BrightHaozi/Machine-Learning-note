import torch
import torch.nn as nn

N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in,requires_grad=True)
y=torch.randn(N,D_out,requires_grad=True)

class TwoLayerNet(nn.Module):
    # init定义层 define the model architecture
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)

    def forward(self, x):
        y_pred=self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayerNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(reduction='sum')

learning_rate=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()