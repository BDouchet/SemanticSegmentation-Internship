import cv2
import numpy as np
import matplotlib.pyplot as plt

### allows to visualize the masks of the image. For instance : [2,3] for class 2 AND 3###

def visualise(filename,filter):
    img = cv2.imread(filename)[:, :, 2]
    mask=np.zeros((np.shape(img)))
    for i in filter:
        mask[:,:][img[:,:]==i]=1
    mask = mask.astype(np.uint8)
    plt.imshow(mask*255)

### To overlay masks on the image. For instance : [2,3] for class 2 AND 3###

def visualise_over_image(img_file,mask_file,filter):
    img = cv2.imread(img_file)
    mask=cv2.imread(mask_file)[:,:,2]
    for i in filter:
        img[:,:][mask[:,:]==i]=[255,0,0]
    plt.imshow(img)
