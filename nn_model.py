import torch
from torch import nn
import torchvision
from torch.nn import Conv2d, MaxPool2d, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)
class Div(nn.Module):
    def __init__(self):
        super(Div,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=0)
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)
        self.relu1 = ReLU()
        self.sigmod1 = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(196608,10)
    def forward(self,x):
        #x = self.conv1(x)
        #x = self.maxpool1(x)
        #x = self.relu1(x)
        #x = self.sigmod1(x)
        x = self.linear1(x)
        return x

writer = SummaryWriter("logs")
step = 0
div = Div()
for data in dataloader:
    img,target = data
    img = torch.flatten(img)
    output = div(img)
    print(output)
    # writer.add_images("in",img,step)
    # writer.add_images("out",output,step)
    step = step + 1
writer.close()