import torch as t
import numpy as np
from scipy import signal
import random

def gauss_kernel(length, std_dev, device):
    k1d = signal.gaussian(length,std=std_dev).reshape(length, 1)
    return t.Tensor(np.outer(k1d, k1d)).to(device)

def unit_square(length, r, device):
    w = np.zeros((length , length))
    r = int((length - r) / 2)
    w[r:-r, r:-r] = 1
    return t.Tensor(w).to(device)

def random_ones(length, n, device):
    w = np.zeros((length , length))
    pts = [divmod(i, length) for i in random.sample(range(length**2), n)]
    for pt in pts:
        w[pt[0]][pt[1]] = 1
    return t.Tensor(w).to(device)

def cle_weights(length, sp, num, device):
    # sp is scanPath,
    # num is number of unique attention points
    # change to 26x26 coords
    st, indd = np.unique(sp, return_index = True, axis=0)
    sp = (sp * (26/224)).astype(int)
    # make arr unique
    spU, ind  = np.unique(sp, return_index=True, axis=0)
    spU = spU[np.argsort(ind)]
    print("shape of unique sp: ",sp.shape , st.shape,spU.shape)
    if spU.shape[0] > num:
        # clip the extra points
        spU = spU[:num]
    #
    w = np.zeros((length , length))
    for i,c in enumerate(spU):
        w[c[1]][c[0]] = 1
    return t.Tensor(w).to(device)
