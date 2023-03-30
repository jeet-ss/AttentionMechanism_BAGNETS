###
# here we try different weighting mechanisms
###

import numpy as np
import torch
import time
import h5py
import pandas as pd
from w_functions import gauss_kernel, unit_square, random_ones, cle_weights
import sys
import csv


if __name__ == '__main__':
    # load the arguments
    args = [int(i) for i in sys.argv[1:]]
    #args = [1, 2]
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("**",device,"**")
    # load the data and labels
    file_path = '/home/woody/iwso/iwso060h/outputs/featureFile1'
    data_file = h5py.File(file_path, 'r')
    features = data_file['features']
    labels = data_file['labels']
    #features = features[:20]
    #labels = labels[:20]
    print("Data Loading Complete, size: ", len(features) )
    # load the saliency map
    f_path = '/home/woody/iwso/iwso060h/outputs/cle_path/all2.hdf5'
    itti_cle = h5py.File(f_path, 'r')
    #itti = itti_cle['smap']
    #itti = itti[:20]
    cle = itti_cle['spath']
    # num of unique pts
    num_points = args[0]
    #cle = cle[:20]
    #
    correct = np.array([])
    top5_accuracy = np.array([])
    st = time.time()
    #
    #value_range = np.arange(10, 12, 1)
    for j in args:
        #weight_matrix = torch.stack([random_ones(features.shape[1], n=j, device=device)]*features.shape[3])[None,:]
        #weight_matrix = torch.stack([unit_square(features.shape[1], r=j, device=device)]*features.shape[3])[None,:]
        #weight_matrix = torch.stack([gauss_kernel(features.shape[1], std_dev=j, device=device)]*features.shape[3])[None,:]
        #print("shape:", weight_matrix.shape)
        #weight_matrix = torch.stack([torch.rand((features.shape[1], features.shape[2]), device=device)]*features.shape[3])[None,:]  # (1,1000,26,26)
        #
        correct1 = 0
        correct5 = 0
        #for i,f1 in enumerate(features):
        for i,(f1,sp) in enumerate(zip(features,cle)):
            # convert to torch
            f1 = torch.Tensor(f1).to(device)
            #sm = torch.from_numpy(sm).to(device)
            #sp = torch.Tensor(sp * (26/224)).type(torch.int64).to(device)
            #
            weight_matrix = cle_weights( f1.shape[1], sp, num = num_points, device = device)
            # increase batch dimension
            f1 = f1[None,:]
            # change dimension for Avgpool2D
            f1 = f1.permute(0,3,1,2)  # (1,1000,26,26)
            # update with the weights
            f1 = f1*weight_matrix
            ##f1 = f1*sm  # for itti
            # weigh with avgpooling
            f1 = torch.nn.AvgPool2d(f1.size()[2], stride=1)(f1).reshape(-1)  # (1,1,1000)
            # find argmax
            best1 = torch.argmax(f1).cpu()
            top5 = torch.topk(f1, 5, dim=0)
            #
            #print(i, ":", labels[i].shape, f1.shape, best1, labels[i])
            #
            if best1 == labels[i]:
                correct1 += 1
            # top 5 accuracy
            if top5.indices.eq(labels[i]).sum().cpu().numpy() > 0:
                correct5 += 1
        #
        acc = correct1/len(features)
        acc5 = correct5/len(features)
        print("Accuracy of our Model: " , acc, acc5 )
        correct = np.append(correct, acc)
        top5_accuracy = np.append(top5_accuracy, acc5)
    print (correct, top5_accuracy, " with time: ", ( time.time() - st ) )
    # for writing one set of values only
    triplet = [args[0], correct[0], top5_accuracy[0]]
    with open('/home/woody/iwso/iwso060h/outputs/cle_result/cle'+str(args[0]), 'w') as fp:
    #with open('/home/woody/iwso/iwso060h/outputs/cle2_result', 'a') as fp:
        wr = csv.writer(fp, delimiter=',',  quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(triplet)
