import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
vgg16 = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
dataset = torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1,drop_last=True)
torch.save(vgg16,"vgg16_mth1.pth")
#vgg16.classifier.add_module("add_linear",nn.Linear(1000,10))
vgg16.classifier[6] = nn.Linear(4096,10)
print(vgg16)