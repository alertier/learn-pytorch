import torch
from torch import nn
import torchvision
from torch.nn import Conv2d, MaxPool2d, ReLU,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
train_data = torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)


train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=64,drop_last=True)
test_dataloader = DataLoader(test_data,batch_size=64,drop_last=True)

writer = SummaryWriter("logs")

total_train_step = 0
total_test_step = 0
epoch = 50
learning_rate = 0.01

loss = nn.CrossEntropyLoss()
loss = loss.cuda()
div = Div()
div = div.cuda()
optim = torch.optim.SGD(div.parameters(),lr=learning_rate)

for i in range(epoch):
    print("---------------第{}轮训练开始---------------".format(i+1))
    run_loss = 0.0

    div.train()
    for data in train_dataloader:
        img,target = data
        img = img.cuda()
        target = target.cuda()
        output = div(img)
        result_loss = loss(output,target)

        optim.zero_grad()
        result_loss.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if(total_train_step % 100 == 0):
            print("第{}次,Loss:{}".format(total_train_step,result_loss.item()))
        writer.add_scalar("train_loss",result_loss.item(),total_train_step)
        run_loss = run_loss + result_loss

    total_test_loss = 0
    total_acc = 0
    div.eval()
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            img = img.cuda()
            target = target.cuda()
            output = div(img)
            result_loss = loss(output, target)
            total_test_loss = total_test_loss + result_loss.item()
            acc = (output.argmax(1) == target).sum()
            total_acc = total_acc + acc

    print("测试的loss:{}".format(total_test_loss))
    print("整体正确率:{}".format(total_acc))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_acc", total_acc/test_data_size, total_test_step)
    total_test_step = total_test_step+1
        # writer.add_images("in",img,step)
        # writer.add_images("out",output,step)
    torch.save(div,"div_{}.pth".format(i))
    print("已保存")
writer.close()