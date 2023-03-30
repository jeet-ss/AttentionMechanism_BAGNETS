###
# this is for checking new model with original bagnets model
###
import numpy as np
import torch
from skimage.io import imread, imread_collection
import torchvision as tv
import bagnets.pytorchnet
import time
import h5py
from dataset import ImgNet_TestDataSet
import pandas as pd


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("**",device,"**")
    # load the data and labels
    file_path = '/home/woody/iwso/iwso060h/outputs/featureFile1'
    data_file = h5py.File(file_path, 'r')
    features = data_file['features']
    labels = data_file['labels']
    #features = features[:1000]
    #labels = labels[:1000]
    #
    file_path2 = '/home/woody/iwso/iwso060h/outputs/outBagnet1'
    data_file2 = h5py.File(file_path2, 'r')
    features2 = data_file2['features']
    #features2 = features2[:1000]
    #
    if len(features) != len(features2):
        print(len(features), len(features2))
        raise Exception("Two sets must be equal length")
    print("Data Loading Complete, size: ", len(features) )
    
    #
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    top5_1 = 0
    top5_2 = 0
    # pass them through a weighing funtion
    for i,(f1, f2) in enumerate(zip(features,features2)):
        
        # convert to torch
        f1 = torch.Tensor(f1).to(device)
        # increase batch dimension
        f1 = f1[None,:]
        # change dimension for Avgpool2D
        f1 = f1.permute(0,3,1,2)
        # weigh with avgpooling
        f1 = torch.nn.AvgPool2d(f1.size()[2], stride=1)(f1).reshape(-1)  # (1,1,1000)
        # fing argmax
        best1_1 = torch.argmax(f1).cpu()
        best2_1 = np.argmax(f2)
        best1_5 = torch.topk(f1, 5, dim=0)
        best2_5 = np.argsort(f2)[-5:]
        #x = best1_5.indices.eq(labels[i]).sum().cpu().numpy()
        #print(x)
        #
        #print(i, ":", labels[i].shape, f1.shape, f2.shape, best1, best2, labels[i])
        #
        flag1 = False
        flag2 = False
        if best1_1 == labels[i]:
            flag1 = True
            correct1 += 1
        if best2_1 == labels[i]:
            flag2 = True
            correct2 += 1
        if flag1 and flag2:
            correct3 += 1
        if flag1==True and flag2==False:
            correct4 += 1
        if flag2==True and flag1==False:
            correct5 += 1
        if best1_5.indices.eq(labels[i]).sum().cpu().numpy() > 0:
            top5_1 += 1
        if labels[i] in best2_5:
            top5_2 += 1
    print("Accuracy of our Model: " , (correct1/len(features)) )
    print("Accuracy of Bagnets Model: ", (correct2/len(features)) )
    print("Top5 Acc of our Model:", (top5_1/len(features)))
    print("Top5 Acc of Original Bagnets:", (top5_2/len(features)))
    print("Percentage when both were correct: ", (correct3/len(features)) )
    print("Percentage when our model performed better: ", (correct4/len(features)) )
    print("Percentage when our model performed worse: ", (correct5/len(features)) )
    # save to file
    file_path = '/home/woody/iwso/iwso060h/outputs/reference'
    with open(file_path, 'w') as fp:
        fp.write("Accuracy of our Model: "+str(correct1/len(features)) )
        fp.write("Top5 Acc of our Model: "+str(top5_1/len(features)) )
    
    '''
    #
    # 2nd PART
    #
    # load the model
    model_path = "/home/woody/iwso/iwso060h/Model/bagnet17_avgFalse_trained_cuda.pt"
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    print("Model Load Complete")
    # load the data
    st = time.time()
    data_path = '/home/vault/iwso/iwso060h/test/ILSVRC2010_test_*.JPEG'
    data_set = imread_collection(data_path)
    print("Data Length: ",len(data_set))
    # prepare bacthes
    batch_size = 20
    test_batches = torch.utils.data.DataLoader(ImgNet_TestDataSet(data_set),
                                               batch_size=batch_size, shuffle=False)
    # pass images through bagnets with avg true
    for i, b in enumerate(test_batches):
        temp = model(b)
        temp.detach_()
    '''
