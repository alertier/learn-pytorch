from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
writer = SummaryWriter("dataset/logs")
image_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = cv.imread(image_path)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
writer.add_image("test",img,7,dataformats="HWC")
# y = x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()