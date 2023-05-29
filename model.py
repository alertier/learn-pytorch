import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear

class Div(nn.Module):
    def __init__(self):
        super(Div,self).__init__()
        self.model1 = torch.nn.Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024,64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    div = Div()
    input = torch.ones((64,3,32,32))
    output = div(input)
    print(output.shape)