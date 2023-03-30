import numpy as np
import torch
from skimage.io import imread, imread_collection
import torchvision as tv


train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]
	
class ImgNet_TestDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = data
        self.transforms = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(size=(224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])
        self.transform2 = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(size=(224, 224)),
            tv.transforms.Grayscale(num_output_channels=3),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        #print("data_size", index, img.shape, img.ndim)
        #img = self.transforms(img).float().to(self.device)
        #print("after", img.shape, img.dim())
        if img.ndim != 3:
            print("small img: ", img.shape)
            img = self.transform2(img).float().to(self.device)
            print("after", img.shape, img.dim())
        else:
            img = self.transforms(img).float().to(self.device)
        #print("after2: ", img.shape, img.dim())
        return img
