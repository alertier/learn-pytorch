import torchvision
from PIL import Image

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./data",train=True,transform=dataset_trans,download=True)
test_set = torchvision.datasets.CIFAR10(root="./data",train=False,transform=dataset_trans,download=True)

