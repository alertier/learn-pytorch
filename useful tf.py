from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("image/70df2bcc7550efe03a50712ca3222704.jpeg").convert('RGB')

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor",img_tensor)

tran_norm = transforms.Normalize([1,3,5],[3,2,1])
img_norm = tran_norm(img_tensor)

writer.add_image("norm",img_norm,1)
writer.close()