"""
Convert processed images and masks in a HDF5 database

Encode masks to one-hot format

Images are in x_train and masks in y_train
"""

import h5py
import os
import cv2
import numpy as np

dir='dataset1700/'
dir_img=dir+'images/'
dir_mask=dir+'masks/'

width=400
height=560
nbr_mask=10

files=os.listdir(dir_img)

hf=h5py.File(dir+'dataset1700R.h5','w')
i=0
hf.create_dataset('x_train', (len(files), height, width, 3), dtype=np.float32)
hf.create_dataset('y_train', (len(files), height, width, nbr_mask), dtype=np.uint8)
for file in files:
    img = cv2.resize(cv2.imread(dir_img + file), (width, height)) / 255
    hf['x_train'][i, ...] = img[None]
    mask = cv2.resize(cv2.imread(dir_mask + file), (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 2]
    mask_1h = np.zeros(shape=(height, width, nbr_mask), dtype=np.uint8)
    for label in range(nbr_mask):
      mask_1h[:, :, label][mask == label] = 1
    hf['y_train'][i, ...] = mask_1h[None]
    i += 1
hf.close()
