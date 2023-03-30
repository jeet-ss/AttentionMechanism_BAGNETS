###
# extract features of size ()
###
import numpy as np
import torch
import time
from skimage.io import imread, imread_collection
import torchvision as tv
from dataset import ImgNet_TestDataSet
import h5py
import pandas as pd

train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("device", device)
    # model
    model_path = "/home/woody/iwso/iwso060h/Model/bagnet17_avgFalse_trained_cuda.pt"
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    # original model
    model_path2 = "/home/woody/iwso/iwso060h/Model/bagnet17_trained.pt"
    model2 = torch.load(model_path2, map_location=torch.device(device))
    model2.eval()
    print("Model Load Complete")

    transforms_ = tv.transforms.Compose([
            #tv.transforms.ToPILImage(),
            tv.transforms.Resize(size=(224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])
    # data
    st = time.time()
    #data_path = '/home/woody/iwso/iwso060h/val/ILSVRC2010_val_*.JPEG'
    #data_set = imread_collection(data_path)
    data_path = '/home/woody/iwso/iwso060h/Images/'
    data_set = tv.datasets.ImageNet(data_path, split='val', transform=transforms_ )
    print("Total Data Length: ",len(data_set), "\n Time Taken: ", (time.time()-st))
    
    #prepare images for batching
    batch_size = 20
    test_batches = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

    # read the labels
    labels_path = '/home/woody/iwso/iwso060h/labels/ILSVRC2010_validation_ground_truth.txt'
    labels_file = pd.read_csv(labels_path, sep=' ')
    labels = labels_file.to_numpy().reshape(-1)
    print("lables: ", labels.shape)
    #
    out_path = '/home/woody/iwso/iwso060h/outputs/featureFile1'
    out_file = h5py.File(out_path, 'w')
    out_data = out_file.create_dataset('features', (0, 26, 26, 1000), maxshape=(None, 26, 26, 1000))
    out_labels = out_file.create_dataset('labels', (0, ), maxshape=(None, ))
    #
    out_path2 = '/home/woody/iwso/iwso060h/outputs/outBagnet1'
    out_file2 = h5py.File(out_path2, 'w')
    out_data2 = out_file2.create_dataset('features', (0, 1000), maxshape=(None, 1000))
    
    # load the labels
    #out_labels[:] = labels
    #
    start_time = time.time()
    for i,b in enumerate(test_batches):
        #print(len(b), b[0].shape, b[1].shape)
        temp = model(b[0].to(device))
        temp.detach_()
        out_data.resize((len(out_data)+batch_size), axis=0)
        out_data[(i*batch_size):((i+1)*batch_size), :, :, :] = temp.cpu().numpy()
        out_labels.resize((len(out_labels)+batch_size), axis=0)
        out_labels[(i*batch_size):((i+1)*batch_size),] = b[1].numpy()
        #
        temp2 = model2(b[0].to(device))
        print("original", temp2.shape)
        temp2.detach_()
        out_data2.resize((len(out_data2)+batch_size), axis=0)
        out_data2[(i*batch_size):((i+1)*batch_size), :] = temp2.cpu().numpy()
        print(i, len(out_data), len(out_data2) )
    print("Time taken: ", (time.time() - start_time))

