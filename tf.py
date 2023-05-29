from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

#tensor数据类型

img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# writer.add_image()

writer.close()