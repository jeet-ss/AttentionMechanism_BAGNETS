##
# This is used to join the multiple files created in CLE
##
import os
import numpy as np
import h5py as h

if __name__ == '__main__':
    path = "/home/woody/iwso/iwso060h/outputs/cle_path/"

    with h.File(path+"all.hdf5", "w") as f_dst:
        h5files = sorted([f for f in os.listdir(path) if f.endswith(".h5")])
        print(h5files)
        
        dset = f_dst.create_dataset("spath", shape=(len(h5files), 10000, 400, 4), dtype='f4')
        
        for i, filename in enumerate(h5files):
            print(i, filename)
            with h.File(path+filename) as f_src:
                dset[i] = f_src['spath']

