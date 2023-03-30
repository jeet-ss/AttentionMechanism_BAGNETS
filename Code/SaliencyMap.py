import numpy as np
import torch
import h5py
import time
import torchvision as tv
from dataset import ImgNet_TestDataSet
#from models.models import get_itti_sm as Itti_Saliency_Map
from models.models import ConstrainedLevyExploration


train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("device", device)
    #
    transforms_ = tv.transforms.Compose([
            #tv.transforms.ToPILImage(),
            tv.transforms.Resize(size=(224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])
    # Data
    st = time.time()
    data_path = '/home/woody/iwso/iwso060h/Images/'
    data_set = tv.datasets.ImageNet(data_path, split='val', transform=transforms_ )
    indices = torch.arange(0,10000)  # 10 means 9
    data_set = torch.utils.data.Subset(data_set, indices)
    print("Total Data Length: ",len(data_set), "\n Time Taken: ", (time.time()-st))
    #prepare images for batching
    batch_size = 1
    test_batches = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # for itti s map
    weight_length = 26
    # for CLE
    sp_length = 400
    #
    out_path = '/home/woody/iwso/iwso060h/outputs/cle_path/0to10000.h5'
    out_file = h5py.File(out_path, 'w')
    #out_smap = out_file.create_dataset('smap', (0, weight_length, weight_length), maxshape=(None, weight_length, weight_length))
    out_spath = out_file.create_dataset('spath', (0, sp_length, 4), maxshape=(None, sp_length, 4))
    # loop over the data
    for i,b in enumerate(test_batches):
        #print(i, b[0][0].shape, b[1].shape, len(b))
        # prepare img
        img = b[0][0].permute(1,2,0).cpu().numpy()
        # itti_sm
        #itti_sm = Itti_Saliency_Map(img)
        #itti_sm = cv2.resize(itti_sm, (weight_length, weight_length))
        #CLE scanpath
        cle_model = ConstrainedLevyExploration(img)
        cle_path = cle_model.scanpath(scanpath_length=sp_length)
        # save to file
        #out_smap.resize((len(out_smap)+batch_size), axis=0)
        #out_smap[(i*batch_size):((i+1)*batch_size), :, :] = itti_sm
        out_spath.resize((len(out_spath)+batch_size), axis=0)
        out_spath[(i*batch_size):((i+1)*batch_size), :, :] = cle_path




